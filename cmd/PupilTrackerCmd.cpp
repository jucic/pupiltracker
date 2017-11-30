#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include <pupiltracker/PupilTracker.h>
#include <pupiltracker/cvx.h>
#include <pupiltracker/utils.h>

void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
  cv::Mat res;
  cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
  cv::imshow(name, res);
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Need filename" << std::endl;
    return 1;
  }

  std::cout << "Opening " << argv[1] << std::endl;
  cv::VideoCapture vc(argv[1]);
  if (!vc.isOpened()) {
    std::cerr << "Could not open " << argv[1] << std::endl;
    return 2;
  }

  pupiltracker::TrackerParams params;
  pupiltracker::ConfigFile cgf;
  cgf.read("../cgf/params.cgf");
  params.Radius_Min = cgf.get<int>("Radius_Min");
  params.Radius_Max = cgf.get<int>("Radius_Max");

  params.CannyBlur = cgf.get<int>("CannyBlue");
  params.CannyThreshold1 = cgf.get<int>("CannyThreshold1");
  params.CannyThreshold2 = cgf.get<int>("CannyThreshold2");
  params.StarburstPoints = cgf.get<int>("StarburstPoints");

  params.PercentageInliers = cgf.get<int>("PercentageInliers");
  params.InlierIterations = cgf.get<int>("InlierIterations");
  params.ImageAwareSupport = cgf.get<bool>("ImageAwareSupport");
  params.EarlyTerminationPercentage = cgf.get<int>("EarlyTerminationPercentage");
  params.EarlyRejection = cgf.get<int>("EarlyRejection");
  params.Seed = cgf.get<int>("Seed");


  cv::Mat m;
  while (true)
  {
    vc >> m;
    if (m.empty())
    {
      vc.release();
      vc.open(argv[1]);
      if (!vc.isOpened()) {
        std::cerr << "Could not open " << argv[1] << std::endl;
        return 2;
      }
      vc >> m;
      if (m.empty()) {
        return 1;
      }
    }


    pupiltracker::findPupilEllipse_out out;
    pupiltracker::tracker_log log;
    pupiltracker::findPupilEllipse(params, m, out, log); 

    pupiltracker::cvx::cross(m, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
    cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255,0,255));

    std::cout << out.pPupil << std::endl;

    //imshowscale("Haar Pupil", out.mHaarPupil, 3);
    //imshowscale("Pupil", out.mPupil, 3);
    //imshowscale("Thresh Pupil", out.mPupilThresh, 3);
    //imshowscale("Edges", out.mPupilEdges, 3);
    //cv::imshow("Result", m);

    //if (cv::waitKey(10) != -1)
    //  break;
  }
}
