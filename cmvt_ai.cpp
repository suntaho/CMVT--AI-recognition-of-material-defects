#include "cmvt_ai.h"

// general parameters
double gamma = 0.2;                                                    // gamma coefficient to enhance image's contrast
char *Cfont_path = "c:\\windows\\fonts\\msjh.ttc";                     // path of chinese fonts
// parameters for histogram
const int hist_inty_target = 128;                                      // the target intensity for adjusting histogram
// parameters for pylon camera
int64_t newWidth = 1024;                                               // set width of pylon image
int64_t newHeight = 768;                                               // set height of pylon image
double exposuretime = 35000.0;                                         // set exposure time(us) of pylon image
// parameters for keypoints
int nkpnt = 100;                                                       // number of keypoints
// parameters for defect of color difference
const int brightness_lng = 128;                                        // length of brightness data series for sequential frames
const int brightness_count = 10;                                       // number of previous frame for calculating average/stable brightness
double clr_diff = 0.07;                                                // define the defect of color difference if brightness variation (%) > clr_diff
// parameters for keypoint clustering
int kdim = 3;                                                          // 2 for x/y; 3 for x/y/brightness for k-means data
int brg_range = 32;                                                    // if kdim=3, the characteristic distance of brightness in k-means method
double brg_thrsd_r = 1.10;                                             // accept the point-cluster if the average brightness of its bounding box > brg_thrsd_r*hist_inty_target(t)
double rect_overlap_dist = 5.0;                                        // the threshold distance (pixel) to group two rectangle into one
bool rotatedbox = false;                                               // detect bounding box by rotated or upright rectangles
int scaleL = 2;                                                        // scale range of rectangle for improving keypoint clustering 
int scaleH = 4;
// parameters for AI
long max_num_training_data = 10000;                                    // maximum number of training data
bool ANN = true;                                                       // true for ANN, false for ML
int ANNlayer = 2;                                                      // number of hidden layers
int ANNneuron = 20;                                                    // number of neurons for each ANN inner layer
int ntype_defect = 6;                                                  // number of defect type
bool plt_t_contour = true;                                             // copy the training image with contours
int intpSize = 15;                                                     // rescaled image size for extract features of AI
// video
bool saveVideo = false;                                                // save AI process into video


int main(int argc, char* argv[])
{
	// initialize camera
	VideoCapture cap;
	bool pyloncamera = false;                                          // usage of pylon-camera
	if (argc > 1 && *argv[1] == 'p') {
		Pylon_ini();                                                   // initialize Pylon camera
		Pylon_setWH(newWidth, newHeight);                              // adjust camera's widht/hgieht
		Pylon_setET(exposuretime);                                     // adjust camera's Exposure-Time(us)
		Pylon_begin();                                                 // start the grabbing of Pylon images
		if (!Pylon_IsGrabSuceed()) {
			cout << "no pylon camera is found!" << endl;
			exit(1);
		}
		pyloncamera = true;
	}
	else { 
		cap.open(0);                                                   // use default camera
		if (!cap.isOpened()) {
			cout << "no default camera is found!" << endl;
			exit(1);
		}
	}

	// build LookUpTable for adjusting gamma of gray image;
	Mat lookUpTable = build_lookUpTable(gamma);         

	// automatically/manually adjust exposure time
	bool manual = false;
	int maxloc_g, int_exposuretime = cvRound(exposuretime/1000.0);
	double auto_factor = 5000.0;
	Mat brightness(1,brightness_lng,CV_64F);                           // brightness data series of sequential frames
	brightness.setTo(cv::Scalar::all(1.0));
	while (char(waitKey(1)) != 27) {
		// grab a camera frame
		Mat frame, frame_gray;
		if (pyloncamera) { if (Pylon_IsGrabSuceed()) frame = Pylon2Mat(); }
		else { cap >> frame; }
		// check if grabbing frame is successful
		if (frame.empty()) {
			cout << "could not grab a camera frame" << endl;
			exit(1);
		}
		// convert to gray-image
		if (frame.channels() == 3) {
			cvtColor(frame, frame_gray, CV_BGR2GRAY);
		}
		else {
			frame_gray = frame.clone();
		}
		frame.release();
		// adjust gamma for gray-image
		LUT(frame_gray, lookUpTable, frame_gray);
		// update brightness series
		Mat m1 = brightness.clone();
		m1(cv::Rect(1, 0, brightness_lng - 1, 1)).copyTo(brightness(cv::Rect(0, 0, brightness_lng - 1, 1)));
		brightness.at<double>(0, brightness_lng - 1) = (double)sum(frame_gray)[0];
		// generate trackbar(ms) to adjust Exposure-Time(us)
		auto_factor = auto_factor*0.90;
		maxloc_g = GUI_ET(frame_gray, brightness, Cfont_path, &int_exposuretime, &exposuretime, &auto_factor, &manual, hist_inty_target);
	}
	if (manual) exposuretime = (double)int_exposuretime*1000.0;
	destroyWindow("Histogram");
	destroyWindow("Brightness");
	destroyWindow("Adjustment for exposure time");
	cout << "adjust exposure time (us): " << exposuretime << endl;
	cout << "the max-accumulated gray level: " << maxloc_g << endl;
	waitKey(500);

	// AI processing...
	int gui_mode = 0;                                                     // 0 for detection; 1 for collection; 2 for recognition
	brightness.setTo(cv::Scalar::all(1.0));
	VideoWriter put("output.mpg", CV_FOURCC('M', 'P', 'E', 'G'), 10, Size(newWidth, newHeight));
	while (char(waitKey(1)) != 27) {
		// grab a camera frame
		Mat frame, frame_gray;
		if (pyloncamera) { if (Pylon_IsGrabSuceed()) frame = Pylon2Mat(); }
		else { cap >> frame; }
		// check if grabbing frame is successful
		if (frame.empty()) {
			cout << "could not grab a camera frame" << endl;
			exit(1);
		}
		// convert to gray-image
		if (frame.channels() == 3) {
			cvtColor(frame, frame_gray, CV_BGR2GRAY);
		}
		else {
			frame_gray = frame.clone();
		}
		frame.release();
		// adjust gamma for gray-image
		LUT(frame_gray, lookUpTable, frame_gray);
		// update brightness series
		Mat m1 = brightness.clone();
		m1(cv::Rect(1, 0, brightness_lng - 1, 1)).copyTo(brightness(cv::Rect(0, 0, brightness_lng - 1, 1)));
		brightness.at<double>(0, brightness_lng - 1) = (double)sum(frame_gray)[0];
		// call GUI
		Mat framei = GUI(frame_gray, brightness, Cfont_path, &gui_mode, nkpnt, brightness_count, clr_diff,
			kdim, brg_range, brg_thrsd_r, rect_overlap_dist, ANN, rotatedbox, ntype_defect, 
			plt_t_contour, intpSize, ANNlayer, ANNneuron, scaleL, scaleH);
		// save video
		if (saveVideo) put << framei;
	}
	destroyWindow("Brightness");
	destroyWindow("CMVT_AI");

	// close camera
	if (pyloncamera) { Pylon_close(); }
	else { cap.release(); }
	return 0;
}

