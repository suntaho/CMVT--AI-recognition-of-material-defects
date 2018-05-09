// c++ library
#define _CRT_SECURE_NO_WARNINGS
#pragma comment( lib, "vfw32.lib" ) 
#pragma comment( lib, "comctl32.lib" ) 
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
//#include <windows.h>
// opencv library
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
// pylon library
#include <pylon/PylonIncludes.h>
#include <pylon/PylonGUI.h>
// pylon-GIGE library
// Setting for using Basler GigE cameras.  
#include <pylon/gige/BaslerGigEInstantCamera.h>  
typedef Pylon::CBaslerGigEInstantCamera Camera_t;
// CvxText library for Chinese fonts
#include "CvxText.h"
using namespace cv;
using namespace ml;
using namespace std;
using namespace Pylon;
using namespace GenApi;
using namespace Basler_GigECameraParams;

// ================== subroutines of pylon_module.cpp ======================
void Pylon_ini();                                                      // initialize Pylon camera	
void Pylon_setWH(int64_t newWidth, int64_t newHeight);                 // adjust camera's widht/hgieht
void Pylon_setET(double exposuretime);                                 // adjust camera's Exposure-Time(us)
void Pylon_begin();                                                    // start the grabbing of Pylon images
bool Pylon_IsGrabSuceed();                                             // request the grabbing-image status 
Mat Pylon2Mat();                                                       // convert Pylon image to OpenCV image
void Pylon_close();                                                    // close device
// ==================== end of Pylon subroutines ===========================


// ================== subroutines of img_module.cpp ========================
Mat build_lookUpTable(double db_gamma);                                // build LookUpTable for adjusting gamma of gray image
Rect drawString(Mat imgi, char *stri, Point posi, Scalar RGBi, float fontsize, char *Cfont_path);  // draw Chinese text into an image and return its bounding rect
Rect drawButton(Mat imgi, char *stri, Point posi, Scalar RGBi, float fontsize, char *Cfont_path);  // draw Chinese-text/button into an image and return the rect
Mat calHist_gray(Mat imgi, int histSize, int gray_min, int gray_max, int *maxloc_g);               // calculate histogram for a gray image, no drawing
Mat drawHist_gray(Mat imgi, int histSize, int gray_min, int gray_max, int *maxloc_g);              // draw histogram for a gray image
Mat drawHist_bgr(Mat imgi, int histSize, int gray_min, int gray_max);                              // draw histogram for a color image
Mat drawSeries(Mat datai, int imgSize);                                                            // draw data series
void on_trackbar_ET(int pos, void *);                                                              // trackbar(ms) to assign Exposure-Time(us)
void on_mouse_pnt(int event, int x, int y, int flag, void *ptr);                                   // get mouse's position as left-button-clicking
vector<vector<Point>> kypnt_cluster(Mat frame_i, int nkpnt, int inty_target,
	int kdim, int brg_range);                                                                      // finding and clustering key-points
vector<Rect> kypnt2rect(Mat in, vector<vector<Point>> kypcluster, int maxloc_g,
	double brg_thrsd_r, double rect_overlap_dist);                                                 // convert keypoint into grouprectangles
Mat gray_img_cluster(Mat in, int cluster_num, int maxloc_g, bool binarized);                       // clusterizing gray-image by k-means
vector<vector<Point>> cal_closed_contours_maxArae(Mat in, vector<Rect> rects, int maxloc_g,        // find closed contourss for each given rect.
	double brg_thrsd_r, double rect_overlap_dist, int scaleL, int scaleH);
vector<Rect> rect_refine_k(Mat in, vector<Rect> rects, int maxloc_g,
	double brg_thrsd_r, double rect_overlap_dist, int scaleL, int scaleH);                         // refine rectangle by k-means method on video gray-frame
vector<RotatedRect> rotated_rect_refine_k(Mat in, vector<Rect> rects, int maxloc_g,                // refine rotated rectangle by k-means method on video gray-frame.
	double brg_thrsd_r, double rect_overlap_dist, int scaleL, int scaleH);                         // NO grouprectangles function, extract maximum-area one only!!!
int save_choice(Mat in, int maxloc_g, int sw);                                                     // save chosen defect into specific directory
// ==================== end of image subroutines ===========================


// ================== subroutines of gui.cpp ===============================
int GUI_ET(Mat in, Mat brightness, char *Cfont_path, int *int_exposuretime,
	double *exposuretime, double *auto_factor, bool *manual, int hist_inty_target);                // automatically/manually adjust Exposure-Time(us), and return max-accumulated gray level
Mat GUI(Mat in, Mat brightness, char *Cfont_path, int *mode, int nkpnt,
	int brightness_count, double clr_diff, int kdim, int brg_range, double brg_thrsd_r,
	double rect_overlap_dist, bool ANN, bool rotatedbox, int ntype_defect,
	bool plt_t_contour, int intpSize, int ANNlayer, int ANNneuron,
	int scaleL, int scaleH);                                                                       // Graphical User Interface for AI
// ======================= end of GUI ======================================


// ============== subroutines of ai_algorithm.cpp ==========================
Mat featurem(Mat in, Rect rect_in, int maxloc_g, int intpSize, double brg_thrsd_r,
	double rect_overlap_dist, int scaleL, int scaleH, int typei, int ntype_defect);                // quantify features of assigned patch in gray-image, as input of ai-training/ai-recognition
bool train_AI(int ntype_defect, bool ANN, int intpSize, double brg_thrsd_r, 
	double rect_overlap_dist, int scaleL, int scaleH, bool plt_t_contour);                         // training data, if data exists
// ==================== end of ai subroutine ===============================