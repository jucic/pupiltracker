#include "pupiltracker/PupilTracker.h"

#include <iostream>

#include <boost/foreach.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/tbb.h>

#include <pupiltracker/cvx.h>

#include <thread>

namespace
{
  struct section_guard
  {
    std::string name;
    pupiltracker::tracker_log& log;
    pupiltracker::timer t;
    section_guard(const std::string& name, pupiltracker::tracker_log& log) : name(name), log(log), t() {  }
    ~section_guard() { log.add(name, t); }
    operator bool() const {return false;}
  };

  inline section_guard make_section_guard(const std::string& name, pupiltracker::tracker_log& log)
  {
    return section_guard(name,log);
  }
}

#define SECTION(A,B) if (const section_guard& _section_guard_ = make_section_guard( A , B )) {} else




class HaarSurroundFeature
{
  public:
    HaarSurroundFeature(int r1, int r2) : r_inner(r1), r_outer(r2)
  {
    //  _________________
    // |        -ve      |
    // |     _______     |
    // |    |   +ve |    |
    // |    |   .   |    |
    // |    |_______|    |
    // |         <r1>    |
    // |_________<--r2-->|

    // Number of pixels in each part of the kernel
    int count_inner = r_inner*r_inner;
    int count_outer = r_outer*r_outer - r_inner*r_inner;

    // Frobenius normalized values
    //
    // Want norm = 1 where norm = sqrt(sum(pixelvals^2)), so:
    //  sqrt(count_inner*val_inner^2 + count_outer*val_outer^2) = 1
    //
    // Also want sum(pixelvals) = 0, so:
    //  count_inner*val_inner + count_outer*val_outer = 0
    //
    // Solving both of these gives:
    //val_inner = std::sqrt( (double)count_outer/(count_inner*count_outer + sq(count_inner)) );
    //val_outer = -std::sqrt( (double)count_inner/(count_inner*count_outer + sq(count_outer)) );

    // Square radius normalised values
    //
    // Want the response to be scale-invariant, so scale it by the number of pixels inside it:
    //  val_inner = 1/count = 1/r_outer^2
    //
    // Also want sum(pixelvals) = 0, so:
    //  count_inner*val_inner + count_outer*val_outer = 0
    //
    // Hence:
    val_inner = 1.0 / (r_inner*r_inner);
    val_outer = -val_inner*count_inner/count_outer;

  }

    double val_inner, val_outer;
    int r_inner, r_outer;
};


template<typename Iterator, typename T>
T parallel_reduce(Iterator first, Iterator last, T init, std::function<void(Iterator, Iterator, T&)> func, std::function<T(const T&, const T&)> reduce) {
  unsigned long const length = std::distance(first, last);

  if (!length)
    return init;

  unsigned long const min_per_thread = 25;
  unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
  unsigned long const hardware_threads = std::thread::hardware_concurrency();

  unsigned long const num_threads =
    std::min<unsigned long>(hardware_threads != 0 ? hardware_threads:2, max_threads);
  unsigned long const block_size = length / num_threads;

  std::vector<T> results(num_threads);
  std::vector<std::thread> threads(num_threads-1);

  Iterator block_start = first;
  for (unsigned long i=0; i < (num_threads-1); ++i) {
    Iterator block_end = block_start;
    std::advance(block_end, block_size);

    threads[i] = std::thread(func, block_start, block_end, std::ref(results[i]));
    block_start = block_end;
  }
  func(block_start, last, results[num_threads - 1]);
  for (auto & it : threads) it.join();

  T result = results[0];
  for (auto it = results.begin() + 1; it != results.end(); ++it) {
    result = reduce(result, *it);
  }
  return result;
}

int HaarResponse(const cv::Mat& src, int radius_min, int radius_max, std::pair<double, cv::Point2f>& min_response, int rstep = 2, int xstep = 4, int ystep = 4) {
  cv::Mat_<int32_t> integral_img;
  int padding = 2 * radius_max;
  cv::Mat pad_img;
  cv::copyMakeBorder(src, pad_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
  cv::integral(pad_img, integral_img);

  cv::Point2f harr_pupil;
  int haar_radius = 0;

  for (int r = radius_min; r < radius_max; r+=rstep) {
    int r_inner = r;
    int r_outer = r*3;
    HaarSurroundFeature feature(r_inner, r_outer);
    std::vector<int> y_vec;
    for (int y = r; y < src.rows - r; y+=ystep) {
      y_vec.push_back(y);
    }
    auto result = parallel_reduce<std::vector<int>::iterator, std::pair<double, cv::Point2f>>
      (y_vec.begin(), y_vec.end(), min_response, [&](std::vector<int>::iterator begin, std::vector<int>::iterator end, std::pair<double,cv::Point2f>& result) {
        result = min_response;
        for (auto it = begin; it != end; ++it) {
          int y = *it;
          int* row1_inner = integral_img[y+padding - r_inner];
          int* row2_inner = integral_img[y+padding + r_inner + 1];
          int* row1_outer = integral_img[y+padding - r_outer];
          int* row2_outer = integral_img[y+padding + r_outer + 1];

          int* p00_inner = row1_inner + r + padding - r_inner;
          int* p01_inner = row1_inner + r + padding + r_inner + 1;
          int* p10_inner = row2_inner + r + padding - r_inner;
          int* p11_inner = row2_inner + r + padding + r_inner + 1;

          int* p00_outer = row1_outer + r + padding - r_outer;
          int* p01_outer = row1_outer + r + padding + r_outer + 1;
          int* p10_outer = row2_outer + r + padding - r_outer;
          int* p11_outer = row2_outer + r + padding + r_outer + 1;

          for (int x = r; x < src.cols - r; x+=xstep) {
            int sum_inner = *p00_inner + *p11_inner - *p01_inner - *p10_inner;
            int sum_outer = *p00_outer + *p11_outer - *p01_outer - *p10_outer - sum_inner;

            double response = feature.val_inner * sum_inner + feature.val_outer * sum_outer;

            if (response < result.first) {
              result.first = response;
              result.second = cv::Point(x, y);
              haar_radius = r;
            }
            p00_inner += xstep;
            p01_inner += xstep;
            p10_inner += xstep;
            p11_inner += xstep;

            p00_outer += xstep;
            p01_outer += xstep;
            p10_outer += xstep;
            p11_outer += xstep;
          }
        }
      },
        [](const std::pair<double, cv::Point2f>& x, const std::pair<double, cv::Point2f>& y) -> std::pair<double,cv::Point2f> {
          if (x.first < y.first)
            return x;
          else
            return y;
        });
    min_response = result;
  }
  return haar_radius;
}


cv::RotatedRect fitEllipse(const std::vector<pupiltracker::EdgePoint>& edgePoints)
{
  std::vector<cv::Point2f> points;
  points.reserve(edgePoints.size());

  BOOST_FOREACH(const pupiltracker::EdgePoint& e, edgePoints)
    points.push_back(e.point);

  return cv::fitEllipse(points);
}


bool pupiltracker::findPupilEllipse(
  const pupiltracker::TrackerParams& params,
  const cv::Mat& m,

  pupiltracker::findPupilEllipse_out& out,
  pupiltracker::tracker_log& log
  )
{
  // --------------------
  // Convert to greyscale
  // --------------------

  cv::Mat_<uchar> mEye;

  SECTION("Grey and crop", log)
  {
    // Pick one channel if necessary, and crop it to get rid of borders
    if (m.channels() == 1)
    {
      mEye = m;
    }
    else if (m.channels() == 3)
    {
      cv::cvtColor(m, mEye, cv::COLOR_BGR2GRAY);
    }
    else if (m.channels() == 4)
    {
      cv::cvtColor(m, mEye, cv::COLOR_BGRA2GRAY);
    }
    else
    {
      throw std::runtime_error("Unsupported number of channels");
    }
  }

  // -----------------------
  // Find best haar response
  // -----------------------

  //             _____________________
  //            |         Haar kernel |
  //            |                     |
  //  __________|______________       |
  // | Image    |      |       |      |
  // |    ______|______|___.-r-|--2r--|
  // |   |      |      |___|___|      |
  // |   |      |          |   |      |
  // |   |      |          |   |      |
  // |   |      |__________|___|______|
  // |   |    Search       |   |
  // |   |    region       |   |
  // |   |                 |   |
  // |   |_________________|   |
  // |                         |
  // |_________________________|
  //

  cv::Point2f pHaarPupil;
  int haarRadius = 0;

  SECTION("Haar responses", log)
  {

    std::pair<double, cv::Point2f> minRadiusResponse(std::numeric_limits<double>::infinity(), cv::Point2f(-1, -1));
    haarRadius = HaarResponse(mEye, 11, 30, minRadiusResponse);
  }
  // Paradoxically, a good Haar fit won't catch the entire pupil, so expand it a bit
  haarRadius = (int)(haarRadius * SQRT_2);

  // ---------------------------
  // Pupil ROI around Haar point
  // ---------------------------
  cv::Rect roiHaarPupil = cvx::roiAround(cv::Point(pHaarPupil.x, pHaarPupil.y), haarRadius);
  cv::Mat_<uchar> mHaarPupil;
  cvx::getROI(mEye, mHaarPupil, roiHaarPupil);

  out.roiHaarPupil = roiHaarPupil;
  out.mHaarPupil = mHaarPupil;

  // --------------------------------------------------
  // Get histogram of pupil region, segment with KMeans
  // --------------------------------------------------

  const int bins = 256;

  cv::Mat_<float> hist;
  SECTION("Histogram", log)
  {
    int channels[] = {0};
    int sizes[] = {bins};
    float range[2] = {0, 256};
    const float* ranges[] = {range};
    cv::calcHist(&mHaarPupil, 1, channels, cv::Mat(), hist, 1, sizes, ranges);
  }

  out.histPupil = hist;

  float threshold;
  SECTION("KMeans", log)
  {
    // Try various candidate centres, return the one with minimal label distance
    float candidate0[2] = {0, 0};
    float candidate1[2] = {128, 255};
    float bestDist = std::numeric_limits<float>::infinity();
    float bestThreshold = std::numeric_limits<float>::quiet_NaN();

    for (int i = 0; i < 2; i++)
    {
      cv::Mat_<uchar> labels;
      float centres[2] = {candidate0[i], candidate1[i]};
      float dist = cvx::histKmeans(hist, 0, 256, 2, centres, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0.0));

      float thisthreshold = (centres[0] + centres[1])/2;
      if (dist < bestDist && boost::math::isnormal(thisthreshold))
      {
        bestDist = dist;
        bestThreshold = thisthreshold;
      }
    }

    if (!boost::math::isnormal(bestThreshold))
    {
      // If kmeans gives a degenerate solution, exit early
      return false;
    }

    threshold = bestThreshold;
  }

  cv::Mat_<uchar> mPupilThresh;
  SECTION("Threshold", log)
  {
    cv::threshold(mHaarPupil, mPupilThresh, threshold, 255, cv::THRESH_BINARY_INV);
  }

  out.threshold = threshold;
  out.mPupilThresh = mPupilThresh;

  // ---------------------------------------------
  // Find best region in the segmented pupil image
  // ---------------------------------------------

  cv::Rect bbPupilThresh;
  cv::RotatedRect elPupilThresh;

  SECTION("Find best region", log)
  {
    cv::Mat_<uchar> mPupilContours = mPupilThresh.clone();
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mPupilContours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.size() == 0)
      return false;

    std::vector<cv::Point>& maxContour = contours[0];
    double maxContourArea = cv::contourArea(maxContour);
    BOOST_FOREACH(std::vector<cv::Point>& c, contours)
    {
      double area = cv::contourArea(c);
      if (area > maxContourArea)
      {
        maxContourArea = area;
        maxContour = c;
      }
    }

    cv::Moments momentsPupilThresh = cv::moments(maxContour);

    bbPupilThresh = cv::boundingRect(maxContour);
    elPupilThresh = cvx::fitEllipse(momentsPupilThresh);

    // Shift best region into eye coords (instead of pupil region coords), and get ROI
    bbPupilThresh.x += roiHaarPupil.x;
    bbPupilThresh.y += roiHaarPupil.y;
    elPupilThresh.center.x += roiHaarPupil.x;
    elPupilThresh.center.y += roiHaarPupil.y;
  }

  out.bbPupilThresh = bbPupilThresh;
  out.elPupilThresh = elPupilThresh;

  // ------------------------------
  // Find edges in new pupil region
  // ------------------------------

  cv::Mat_<uchar> mPupil, mPupilOpened, mPupilBlurred, mPupilEdges;
  cv::Mat_<float> mPupilSobelX, mPupilSobelY;
  cv::Rect bbPupil;
  cv::Rect roiPupil = cvx::roiAround(cv::Point(elPupilThresh.center.x, elPupilThresh.center.y), haarRadius);
  SECTION("Pupil preprocessing", log)
  {
    const int padding = 3;

    cv::Rect roiPadded(roiPupil.x-padding, roiPupil.y-padding, roiPupil.width+2*padding, roiPupil.height+2*padding);
    // First get an ROI around the approximate pupil location
    cvx::getROI(mEye, mPupil, roiPadded, cv::BORDER_REPLICATE);

    cv::Mat morphologyDisk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mPupil, mPupilOpened, cv::MORPH_OPEN, morphologyDisk, cv::Point(-1,-1), 2);

    if (params.CannyBlur > 0)
    {
      cv::GaussianBlur(mPupilOpened, mPupilBlurred, cv::Size(), params.CannyBlur);
    }
    else
    {
      mPupilBlurred = mPupilOpened;
    }

    cv::Sobel(mPupilBlurred, mPupilSobelX, CV_32F, 1, 0, 3);
    cv::Sobel(mPupilBlurred, mPupilSobelY, CV_32F, 0, 1, 3);

    cv::Canny(mPupilBlurred, mPupilEdges, params.CannyThreshold1, params.CannyThreshold2);

    cv::Rect roiUnpadded(padding,padding,roiPupil.width,roiPupil.height);
    mPupil = cv::Mat(mPupil, roiUnpadded);
    mPupilOpened = cv::Mat(mPupilOpened, roiUnpadded);
    mPupilBlurred = cv::Mat(mPupilBlurred, roiUnpadded);
    mPupilSobelX = cv::Mat(mPupilSobelX, roiUnpadded);
    mPupilSobelY = cv::Mat(mPupilSobelY, roiUnpadded);
    mPupilEdges = cv::Mat(mPupilEdges, roiUnpadded);

    bbPupil = cvx::boundingBox(mPupil);
  }

  out.roiPupil = roiPupil;
  out.mPupil = mPupil;
  out.mPupilOpened = mPupilOpened;
  out.mPupilBlurred = mPupilBlurred;
  out.mPupilSobelX = mPupilSobelX;
  out.mPupilSobelY = mPupilSobelY;
  out.mPupilEdges = mPupilEdges;

  // -----------------------------------------------
  // Get points on edges, optionally using starburst
  // -----------------------------------------------

  std::vector<cv::Point2f> edgePoints;

  if (params.StarburstPoints > 0)
  {
    SECTION("Starburst", log)
    {
      // Starburst from initial pupil approximation, stopping when an edge is hit.
      // Collect all edge points into a vector

      // The initial pupil approximations are:
      //    Centre of mass of thresholded region
      //    Halfway along the major axis (calculated form second moments) in each direction

      tbb::concurrent_vector<cv::Point2f> edgePointsConcurrent;

      cv::Vec2f elPupil_majorAxis = cvx::majorAxis(elPupilThresh);
      std::vector<cv::Point2f> centres;
      centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y));
      centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y) + cv::Point2f(elPupil_majorAxis));
      centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y) - cv::Point2f(elPupil_majorAxis));

      BOOST_FOREACH(const cv::Point2f& centre, centres) {
        tbb::parallel_for(0, params.StarburstPoints, [&] (int i) {
          double theta = i * 2*PI/params.StarburstPoints;

          // Initialise centre and direction vector
          cv::Point2f pDir((float)std::cos(theta), (float)std::sin(theta));  

          int t = 1;
          cv::Point p = centre + (t * pDir);
          while(p.inside(bbPupil))
          {
            uchar val = mPupilEdges(p);

            if (val > 0)
            {
              float dx = mPupilSobelX(p);
              float dy = mPupilSobelY(p);

              float cdirx = p.x - (elPupilThresh.center.x - roiPupil.x);
              float cdiry = p.y - (elPupilThresh.center.y - roiPupil.y);

              // Check edge direction
              double dirCheck = dx*cdirx + dy*cdiry;

              if (dirCheck > 0)
              {
                // We've hit an edge
                edgePointsConcurrent.push_back(cv::Point2f(p.x + 0.5f, p.y + 0.5f));
                break;
              }
            }

            ++t;
            p = centre + (t * pDir);
          }
        });
      }

      edgePoints = std::vector<cv::Point2f>(edgePointsConcurrent.begin(), edgePointsConcurrent.end());


      // Remove duplicate edge points
      std::sort(edgePoints.begin(), edgePoints.end(), [] (const cv::Point2f& p1, const cv::Point2f& p2) -> bool {
        if (p1.x == p2.x)
          return p1.y < p2.y;
        else
          return p1.x < p2.x;
      });
      edgePoints.erase( std::unique( edgePoints.begin(), edgePoints.end() ), edgePoints.end() );

      if (edgePoints.size() < params.StarburstPoints/2)
        return false;
    }
  }
  else
  {
    SECTION("Non-zero value finder", log)
    {
      for(int y = 0; y < mPupilEdges.rows; y++)
      {
        uchar* val = mPupilEdges[y];
        for(int x = 0; x < mPupilEdges.cols; x++, val++)
        {
          if(*val == 0)
            continue;

          edgePoints.push_back(cv::Point2f(x + 0.5f, y + 0.5f));
        }
      }
    }
  }


  // ---------------------------
  // Fit an ellipse to the edges
  // ---------------------------

  cv::RotatedRect elPupil;
  std::vector<cv::Point2f> inliers;
  SECTION("Ellipse fitting", log)
  {
    // Desired probability that only inliers are selected
    const double p = 0.999;
    // Probability that a point is an inlier
    double w = params.PercentageInliers/100.0;
    // Number of points needed for a model
    const int n = 5;

    if (params.PercentageInliers == 0)
      return false;

    if (edgePoints.size() >= n) // Minimum points for ellipse
    {
      // RANSAC!!!

      double wToN = std::pow(w,n);
      int k = static_cast<int>(std::log(1-p)/std::log(1 - wToN)  + 2*std::sqrt(1 - wToN)/wToN);

      out.ransacIterations = k;

      log.add("k", k);

      //size_t threshold_inlierCount = std::max<size_t>(n, static_cast<size_t>(out.edgePoints.size() * 0.7));

      // Use TBB for RANSAC
      struct EllipseRansac_out {
        std::vector<cv::Point2f> bestInliers;
        cv::RotatedRect bestEllipse;
        double bestEllipseGoodness;
        int earlyRejections;
        bool earlyTermination;

        EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
      };
      struct EllipseRansac {
        const TrackerParams& params;
        const std::vector<cv::Point2f>& edgePoints;
        int n;
        const cv::Rect& bb;
        const cv::Mat_<float>& mDX;
        const cv::Mat_<float>& mDY;
        int earlyRejections;
        bool earlyTermination;

        EllipseRansac_out out;

        EllipseRansac(
          const TrackerParams& params,
          const std::vector<cv::Point2f>& edgePoints,
          int n,
          const cv::Rect& bb,
          const cv::Mat_<float>& mDX,
          const cv::Mat_<float>& mDY)
          : params(params), edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), earlyTermination(false), earlyRejections(0)
        {
        }

        EllipseRansac(EllipseRansac& other, tbb::split)
          : params(other.params), edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections)
        {
          //std::cout << "Ransac split" << std::endl;
        }

        void operator()(const tbb::blocked_range<size_t>& r)
        {
          if (out.earlyTermination)
            return;
          //std::cout << "Ransac start (" << (r.end()-r.begin()) << " elements)" << std::endl;
          for( size_t i=r.begin(); i!=r.end(); ++i )
          {
            // Ransac Iteration
            // ----------------
            std::vector<cv::Point2f> sample;
            if (params.Seed >= 0)
              sample = randomSubset(edgePoints, n, static_cast<unsigned int>(i + params.Seed));
            else
              sample = randomSubset(edgePoints, n);

            cv::RotatedRect ellipseSampleFit = fitEllipse(sample);
            // Normalise ellipse to have width as the major axis.
            if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
            {
              ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
              std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
            }

            cv::Size s = ellipseSampleFit.size;
            // Discard useless ellipses early
            if (!ellipseSampleFit.center.inside(bb)
                || s.height > params.Radius_Max*2
                || s.width > params.Radius_Max*2
                || s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
                || s.height > 4*s.width
                || s.width > 4*s.height
               )
            {
              // Bad ellipse! Go to your room!
              continue;
            }

            // Use conic section's algebraic distance as an error measure
            ConicSection conicSampleFit(ellipseSampleFit);

            // Check if sample's gradients are correctly oriented
            if (params.EarlyRejection)
            {
              bool gradientCorrect = true;
              BOOST_FOREACH(const cv::Point2f& p, sample)
              {
                cv::Point2f grad = conicSampleFit.algebraicGradientDir(p);
                float dx = mDX(cv::Point(p.x, p.y));
                float dy = mDY(cv::Point(p.x, p.y));

                float dotProd = dx*grad.x + dy*grad.y;

                gradientCorrect &= dotProd > 0;
              }
              if (!gradientCorrect)
              {
                out.earlyRejections++;
                continue;
              }
            }

            // Assume that the sample is the only inliers

            cv::RotatedRect ellipseInlierFit = ellipseSampleFit;
            ConicSection conicInlierFit = conicSampleFit;
            std::vector<cv::Point2f> inliers, prevInliers;

            // Iteratively find inliers, and re-fit the ellipse
            for (int i = 0; i < params.InlierIterations; ++i)
            {
              // Get error scale for 1px out on the minor axis
              cv::Point2f minorAxis(-std::sin(PI/180.0*ellipseInlierFit.angle), std::cos(PI/180.0*ellipseInlierFit.angle));
              cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height/2 + 1)*minorAxis;
              float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
              float errorScale = 1.0f/errOf1px;

              // Find inliers
              inliers.reserve(edgePoints.size());
              const float MAX_ERR = 2;
              BOOST_FOREACH(const cv::Point2f& p, edgePoints)
              {
                float err = errorScale*conicInlierFit.distance(p);

                if (err*err < MAX_ERR*MAX_ERR)
                  inliers.push_back(p);
              }

              if (inliers.size() < n) {
                inliers.clear();
                continue;
              }

              // Refit ellipse to inliers
              ellipseInlierFit = fitEllipse(inliers);
              conicInlierFit = ConicSection(ellipseInlierFit);

              // Normalise ellipse to have width as the major axis.
              if (ellipseInlierFit.size.height > ellipseInlierFit.size.width)
              {
                ellipseInlierFit.angle = std::fmod(ellipseInlierFit.angle + 90, 180);
                std::swap(ellipseInlierFit.size.height, ellipseInlierFit.size.width);
              }
            }
            if (inliers.empty())
              continue;

            // Discard useless ellipses again
            s = ellipseInlierFit.size;
            if (!ellipseInlierFit.center.inside(bb)
                || s.height > params.Radius_Max*2
                || s.width > params.Radius_Max*2
                || s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
                || s.height > 4*s.width
                || s.width > 4*s.height
               )
            {
              // Bad ellipse! Go to your room!
              continue;
            }

            // Calculate ellipse goodness
            double ellipseGoodness = 0;
            if (params.ImageAwareSupport)
            {
              BOOST_FOREACH(cv::Point2f& p, inliers)
              {
                cv::Point2f grad = conicInlierFit.algebraicGradientDir(p);
                float dx = mDX(p);
                float dy = mDY(p);

                double edgeStrength = dx*grad.x + dy*grad.y;

                ellipseGoodness += edgeStrength;
              }
            }
            else
            {
              ellipseGoodness = inliers.size();
            }

            if (ellipseGoodness > out.bestEllipseGoodness)
            {
              std::swap(out.bestEllipseGoodness, ellipseGoodness);
              std::swap(out.bestInliers, inliers);
              std::swap(out.bestEllipse, ellipseInlierFit);

              // Early termination, if 90% of points match
              if (params.EarlyTerminationPercentage > 0 && out.bestInliers.size() > params.EarlyTerminationPercentage*edgePoints.size()/100)
              {
                earlyTermination = true;
                break;
              }
            }

          }
          //std::cout << "Ransac end" << std::endl;
        }

        void join(EllipseRansac& other)
        {
          //std::cout << "Ransac join" << std::endl;
          if (other.out.bestEllipseGoodness > out.bestEllipseGoodness)
          {
            std::swap(out.bestEllipseGoodness, other.out.bestEllipseGoodness);
            std::swap(out.bestInliers, other.out.bestInliers);
            std::swap(out.bestEllipse, other.out.bestEllipse);
          }
          out.earlyRejections += other.out.earlyRejections;
          earlyTermination |= other.earlyTermination;

          out.earlyTermination = earlyTermination;
        }
      };

      EllipseRansac ransac(params, edgePoints, n, bbPupil, out.mPupilSobelX, out.mPupilSobelY);
      try
      { 
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,k,k/8), ransac);
      }
      catch (std::exception& e)
      {
        const char* c = e.what();
        std::cerr << e.what() << std::endl;
      }
      inliers = ransac.out.bestInliers;
      log.add("goodness", ransac.out.bestEllipseGoodness);

      out.earlyRejections = ransac.out.earlyRejections;
      out.earlyTermination = ransac.out.earlyTermination;


      cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
      ConicSection conicBestFit(ellipseBestFit);
      BOOST_FOREACH(const cv::Point2f& p, edgePoints)
      {
        cv::Point2f grad = conicBestFit.algebraicGradientDir(p);
        float dx = out.mPupilSobelX(p);
        float dy = out.mPupilSobelY(p);

        out.edgePoints.push_back(EdgePoint(p, dx*grad.x + dy*grad.y));
      }

      elPupil = ellipseBestFit;
      elPupil.center.x += roiPupil.x;
      elPupil.center.y += roiPupil.y;
    }

    if (inliers.size() == 0)
      return false;

    cv::Point2f pPupil = elPupil.center;

    out.pPupil = pPupil;
    out.elPupil = elPupil;
    out.inliers = inliers;

    return true;
  }

  return false;
}
