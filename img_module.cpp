#include "cmvt_ai.h"

extern long max_num_training_data;
extern int mouse_pnt_x, mouse_pnt_y;

// build lookUpTable for adjusting gamma of gray image
Mat build_lookUpTable(double db_gamma) {
	Mat lookUpTable0(1, 256, CV_8U);
	uchar* p = lookUpTable0.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, 1.0 / db_gamma) * 255.0);
	return lookUpTable0;
};

// draw Chinese text into an image and return its bounding rect
Rect drawString(Mat imgi, char *stri, Point posi, Scalar RGBi, float fontsize, char *Cfont_path) {
	// initialize Chinese fonts
	CvxText text(Cfont_path);
	float p = 1.0;
	CvScalar *size = &CvScalar(fontsize);
	text.setFont(NULL, size, NULL, &p);
	// Get the bounding box around the text.
	int minx = imgi.cols, miny = imgi.rows, maxx = -1, maxy = -1;
	Mat img_tmp = cv::Mat::zeros(imgi.rows, imgi.cols, CV_8UC3), img_gray;
	IplImage *img2 = &IplImage(img_tmp);
	text.putText(img2, stri, posi, Scalar(255,255,255));
	cvtColor(img_tmp, img_gray, CV_BGR2GRAY);
	for (int k = 0; k < img_gray.rows; k++) {
		const uchar* inData = img_gray.ptr<uchar>(k);
		for (int i = 0; i < img_gray.cols; i++) {
			if (inData[i] == 255) {
				if (minx > i) minx = i;
				if (maxx < i) maxx = i;
				if (miny > k) miny = k;
				if (maxy < k) maxy = k;
			}
		}
	}
	Rect boundingRect(minx - 1, miny - 1, maxx - minx + 2, maxy - miny + 2);
	// Draw anti-aliased text.
	IplImage *img = &IplImage(imgi);
	text.putText(img, stri, posi, RGBi);
	return boundingRect;
}

// draw Chinese-text/button into an image and return the rect
Rect drawButton(Mat imgi, char *stri, Point posi, Scalar RGBi, float fontsize, char *Cfont_path) {
	// initialize Chinese fonts
	CvxText text(Cfont_path);
	float p = 1.0;
	CvScalar *size = &CvScalar(fontsize);
	text.setFont(NULL, size, NULL, &p);
	// Get the bounding box around the text.
	int minx = imgi.cols, miny = imgi.rows, maxx = -1, maxy = -1;
	Mat img_tmp = cv::Mat::zeros(imgi.rows, imgi.cols, CV_8UC3), img_gray;
	IplImage *img2 = &IplImage(img_tmp);
	text.putText(img2, stri, posi, Scalar(255, 255, 255));
	cvtColor(img_tmp, img_gray, CV_BGR2GRAY);
	for (int k = 0; k < img_gray.rows; k++) {
		const uchar* inData = img_gray.ptr<uchar>(k);
		for (int i = 0; i < img_gray.cols; i++) {
			if (inData[i] == 255) {
				if (minx > i) minx = i;
				if (maxx < i) maxx = i;
				if (miny > k) miny = k;
				if (maxy < k) maxy = k;
			}
		}
	}
	Rect boundingRect(minx - 3, miny - 3, maxx - minx + 7, maxy - miny + 7);
	// draw button
	Rect rcbutton(minx - 3, miny - 3, maxx - minx + 7, maxy - miny + 7);
	Mat matbutton = imgi(rcbutton);
	matbutton += Scalar(80, 80, 80);
	rectangle(imgi, rcbutton, Scalar(200, 200, 200), 1, LINE_AA);
	// Draw anti-aliased text.
	IplImage *img = &IplImage(imgi);
	text.putText(img, stri, posi, RGBi);
	return boundingRect;
}

// calculate histogram for a gray image, no drawing
Mat calHist_gray(Mat imgi, int histSize, int gray_min, int gray_max, int *maxloc_g) {
	Mat img = imgi.clone(), gray_hist;
	// set range
	float range[] = { (float)gray_min,(float)gray_max };
	const float* histRange = { range };
	// compute the histogram
	bool uniform = true, accumulate = false;
	calcHist(&img, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);
	// find max-accumulated gray level
	minMaxIdx(gray_hist, NULL, NULL, NULL, maxloc_g);
	return gray_hist;
}

// draw histogram for a gray image
Mat drawHist_gray(Mat imgi, int histSize, int gray_min, int gray_max, int *maxloc_g) {
	Mat img = imgi.clone(), gray_hist;
	// set range
	float range[] = { (float)gray_min,(float)gray_max };
	const float* histRange = { range };
	// compute the histogram
	bool uniform = true, accumulate = false;
	calcHist(&img, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);
	// draw the histogram
	int hist_w = 256, hist_h = 256;                                    // image's widht/height
	int bin_w = cvRound((double)hist_w / histSize);                    // bin's width in histogram
	Mat img_hist(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// normalize the value to [0, img_hist.rows]
	normalize(gray_hist, gray_hist, 0, img_hist.rows, NORM_MINMAX, -1, Mat());
	// draw and find the max-accumulatd gray
	minMaxIdx(gray_hist, NULL, NULL, NULL, maxloc_g);
	for (int i = 1; i < histSize; i++) {
		line(img_hist, Point(bin_w*(i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i))), 
			Scalar(255, 255, 255), 2, 8, 0);
	}
	return img_hist;
}

// draw histogram for a color image
Mat drawHist_bgr(Mat imgi, int histSize, int gray_min, int gray_max) {
	Mat img = imgi.clone(), r_hist, g_hist, b_hist;
	vector<Mat> bgr_planes;
	split(img, bgr_planes);                                            // split r/g/b channels
	// set range
	float range[] = { (float)gray_min,(float)gray_max };
	const float* histRange = { range };
	// compute the histogram
	bool uniform = true, accumulate = false;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	// draw the histogram
	int hist_w = 256, hist_h = 256;                                    // image's widht/height
	int bin_w = cvRound((double)hist_w / histSize);                    // bin's width in histogram
	Mat img_hist(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// normalize the value to [0, img_hist.rows]
	normalize(b_hist, b_hist, 0, img_hist.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, img_hist.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, img_hist.rows, NORM_MINMAX, -1, Mat());
	// draw
	for (int i = 1; i < histSize; i++) {
		line(img_hist, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(img_hist, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(img_hist, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	return img_hist;
}

// drawSeries
Mat drawSeries(Mat datai, int imgSize) {
	Mat data = datai.clone();
	Mat img_data(imgSize, imgSize, CV_8UC3, Scalar(0, 0, 0));
	double maxval;
	minMaxIdx(data, NULL, &maxval, NULL, NULL);
	double step = 1.0*imgSize / data.cols;
	// imaging
	data.at<double>(0, 0) = data.at<double>(0, 0)*img_data.rows / (maxval*1.15);
	for (int i = 1; i < data.cols; i++) {
		data.at<double>(0, i) = data.at<double>(0, i)*img_data.rows / (maxval*1.15);
		line(img_data, Point(cvRound((i - 1)*step), imgSize - cvRound(data.at<double>(0, i - 1))),
			Point(cvRound(i*step), imgSize - cvRound(data.at<double>(0, i))), 
			Scalar(255, 255, 255), 2, 8, 0);
	}
	return img_data;
}

// trackbar to assign Exposure-Time(us)
void on_trackbar_ET(int posx, void *) {}

// get mouse's position as left-button-clicking
void on_mouse_pnt(int event, int x, int y, int flag, void *ptr) {
	if (event == CV_EVENT_LBUTTONDOWN) {
		mouse_pnt_x = x;
		mouse_pnt_y = y;
	}
}

// finding and clustering keypoints
vector<vector<Point>> kypnt_cluster(Mat frame_i, int nkpnt, int inty_target, int kdim, int brg_range) {
	vector<vector<Point>> kpntcluster;
	// keypoint detector and descriptor by ORB method
	Mat frame_ORB, frame_ORB_desc;
	vector<KeyPoint> kp_ORB;
	frame_ORB = frame_i.clone();
	cv::Ptr<cv::Feature2D> detector_ORB = ORB::create(nkpnt);
	cv::Ptr<cv::DescriptorMatcher> descriptor_ORB = DescriptorMatcher::create("BruteForce-Hamming");
	// detect keypoint by BRISK
	GaussianBlur(frame_ORB, frame_ORB, Size(7, 7), 1.0, 1.0);
	detector_ORB->detectAndCompute(frame_ORB, noArray(), kp_ORB, frame_ORB_desc);
	if (kp_ORB.size() == 0) return kpntcluster;
	// assign keypoint's coordinates (and/or its pixel-brightness)
	Mat pnt((int)kp_ORB.size(), kdim, CV_32F), label;
	for (int i = 0; i < kp_ORB.size(); i++) {
		float px, py;
		px = kp_ORB[i].pt.x;
		py = kp_ORB[i].pt.y;
		pnt.at<float>(i, 0) = (float)px;
		pnt.at<float>(i, 1) = (float)py;
		if (kdim == 3)
			pnt.at<float>(i, 2) = (float)(1.0*frame_ORB.at<uchar>(cvRound(py), cvRound(px))*brg_range / inty_target);
	}
	// k-mean method
	int ncluster = cvRound(0.1*kp_ORB.size());
	double varmin = 1.0e12;
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 1.0);
	if (ncluster == 0 && kp_ORB.size() > 0) ncluster = 1;
	for (int clusterCount = 1; clusterCount <= ncluster; clusterCount++) {
		Mat label_try;
		double var = kmeans(pnt, clusterCount, label_try, criteria, 10, KMEANS_PP_CENTERS);
		if (varmin > var) {
			varmin = var;
			label.release();
			label = label_try.clone();
		}
	}
	// sort keypoint cluster
	if (label.rows > 0) {
		double ncluster;
		minMaxIdx(label, NULL, &ncluster, NULL, NULL);
		for (int i = 0; i <= ncluster; i++) {
			vector<Point> contouri;
			for (int j = 0; j < pnt.rows; j++) {
				if (label.at<int>(j, 0) == i)
					contouri.push_back(Point(cvRound(pnt.at<float>(j, 0)), cvRound(pnt.at<float>(j, 1))));
			}
			if (contouri.size() > 1) kpntcluster.push_back(contouri);
		}
	}
	return kpntcluster;;
}

// convert keypoint into grouprectangles
vector<Rect> kypnt2rect(Mat in, vector<vector<Point>> kypcluster, int maxloc_g, double brg_thrsd_r, double rect_overlap_dist) {
	Mat frame_g = in.clone();
	vector<Rect> rects;
	for (int i = 0; i < kypcluster.size(); i++) {
		Rect recti = boundingRect(kypcluster[i]);
		if (sum(frame_g(recti))[0] / recti.width / recti.height > brg_thrsd_r*maxloc_g) {
			rects.push_back(recti);
			rects.push_back(recti);                                    // additional/repeated elements to avoid the loss of isolate rectangle.
		}
	}
	groupRectangles(rects, 1, rect_overlap_dist);
	return rects;
}

// clusterizing gray-image by k-means
Mat gray_img_cluster(Mat in, int cluster_num, int maxloc_g, bool binarized) {
	if (cluster_num <= 1) cluster_num = 2;                             // to avoid numerical divergence
	// initialize variables
	Mat frame_g = in.clone();
	unsigned long int size = frame_g.rows*frame_g.cols;
	Mat pnts(size, 1, CV_32F), label(size, 1, CV_32S), center(cluster_num, 1, CV_32F);
	// transfer gray-level data from frame_g to pnts
	float* pnts_ptr = (float*)pnts.data;
	uchar* frame_ptr = (uchar*)frame_g.data;
	for (unsigned long int i = 0; i < size; i++) {
		*pnts_ptr = (float)(abs(*frame_ptr - 1.0*maxloc_g));
		pnts_ptr++;
		frame_ptr++;
	}
	// k-means method
	Mat out(frame_g.size(), CV_8U, Scalar(0));
	double compactness = kmeans(pnts, cluster_num, label,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 1.0), 10, KMEANS_PP_CENTERS, center);
	if (compactness > pow(1.0, -12.0)) {
		normalize(center, center, 0, 225, NORM_MINMAX, -1, Mat());
		// generate clustering-image
		int* label_ptr = (int*)label.data;
		uchar* out_ptr = (uchar*)out.data;
		for (unsigned long int i = 0; i < size; i++) {
			*out_ptr = (uchar)cvRound(center.at<float>(*label_ptr, 0));
			out_ptr++;
			label_ptr++;
		}
		// if binarized, return maximum-center-value cluster, else return all-level clusters
		if (binarized) {
			// sort centers
			Mat center_tmp = center.clone();
			for (int i = 0; i < center_tmp.rows - 1; i++) {
				for (int j = i + 1; j < center_tmp.rows; j++) {
					if (center_tmp.at<float>(i, 0) > center_tmp.at<float>(j, 0)) {
						float k = center_tmp.at<float>(j, 0);
						center_tmp.at<float>(j, 0) = center_tmp.at<float>(i, 0);
						center_tmp.at<float>(i, 0) = k;
					}
				}
			}
			// find threshold value
			int thresh = cvRound(0.5*(center_tmp.at<float>(center_tmp.rows - 1, 0) + center_tmp.at<float>(center_tmp.rows - 2, 0)));
			threshold(out, out, thresh, 255, THRESH_BINARY);
		}
	}
	return out;
}

// find closed contourss for each given rect.
vector<vector<Point>> cal_closed_contours_maxArae(Mat in, vector<Rect> rects, int maxloc_g,
	double brg_thrsd_r, double rect_overlap_dist, int scaleL, int scaleH) {
	vector<vector<Point>> contours_closed_maxArea;
	Mat frame_g = in.clone();
	for (int i = 0; i < rects.size(); i++) {
		// scale up the rectangle to ensure that it includes the defect
		for (int sl = scaleL; sl <= scaleH; sl++) {
			int x = rects[i].x, y = rects[i].y;
			int width = rects[i].width, height = rects[i].height;
			x = x - cvRound(0.5*(sl - 1)*width);
			y = y - cvRound(0.5*(sl - 1)*height);
			width = sl*width;
			height = sl*height;
			if (x < 0)x = 0;
			if (y < 0)y = 0;
			if ((x + width) >= frame_g.cols) width = frame_g.cols - x - 1;
			if ((y + height) >= frame_g.rows) height = frame_g.rows - y - 1;
			// find contours
			vector<vector<Point>> contours, contours_closed;
			Mat img_local = frame_g(cv::Rect(x, y, width, height));
			Mat img_bin = gray_img_cluster(img_local, 3, maxloc_g, true);                          // img_bin(0,255)
			Mat edge;
			int thresh2 = 2 * maxloc_g;
			if (thresh2 > 235) thresh2 = 235;
			dilate(img_bin, img_bin, Mat(), Point(-1, -1), 2);
			GaussianBlur(img_bin, img_bin, Size(7, 7), 1.0, 1.0);
			Canny(img_bin, edge, maxloc_g, thresh2);
			findContours(edge, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for (int j = 0; j < contours.size(); j++) {
				vector<Point> contourj = contours[j];
				if (contourj.size() > 2) {
					if (abs(contourj[0].x - contourj[contourj.size() - 1].x) < 1.1 && abs(contourj[0].y - contourj[contourj.size() - 1].y) < 1.1)
						contours_closed.push_back(contourj);
				}
			}
			// find maximum-area contour and shift its x,y back to the coordinate of global image
			if (contours_closed.size() > 0) {
				// find the contour having maximum area
				Mat areas((int)contours_closed.size(), 1, CV_64F);
				for (int j = 0; j < (int)contours_closed.size(); j++) {
					areas.at<double>(j, 0) = contourArea(contours_closed[j]);
				}
				int lmax;
				minMaxIdx(areas, NULL, NULL, NULL, &lmax);
				for (int k = 0; k < contours_closed[lmax].size(); k++) {
					contours_closed[lmax][k].x = contours_closed[lmax][k].x + x;
					contours_closed[lmax][k].y = contours_closed[lmax][k].y + y;
				}
				contours_closed_maxArea.push_back(contours_closed[lmax]);
			}
		}
	}
	return contours_closed_maxArea;
}

// refine rectangle by k-means method on video gray-frame
// not use brg_thrsd_r, since average effects at large patch will skip defects
vector<Rect> rect_refine_k(Mat in, vector<Rect> rects, int maxloc_g, double brg_thrsd_r, 
	double rect_overlap_dist, int scaleL, int scaleH) {
	vector<Rect> rects_rf;
	vector<vector<Point>> contours = cal_closed_contours_maxArae(in, rects, maxloc_g, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH);
	for (int i = 0; i < contours.size(); i++) {
		Rect recti = boundingRect(contours[i]);
		rects_rf.push_back(recti);                                     // add two times for safety groupRectangle
		rects_rf.push_back(recti);
	}
	groupRectangles(rects_rf, 1, rect_overlap_dist);
	return rects_rf;
}

// refine rotated-rectangle by k-means method on video gray-frame
// NO grouprectangles function, extract maximum-area one only!!!
// not use brg_thrsd_r, since average effects at large patch will skip defects
vector<RotatedRect> rotated_rect_refine_k(Mat in, vector<Rect> rects, int maxloc_g, double brg_thrsd_r,
	double rect_overlap_dist, int scaleL, int scaleH) {
	vector<RotatedRect> rects_rf;
	for (int i = 0; i < rects.size(); i++) {
		vector<Rect> rects_i;
		rects_i.push_back(rects[i]);
		vector<vector<Point>> contours = cal_closed_contours_maxArae(in, rects_i, maxloc_g, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH);		
		// find the rotated-rectangle having maximum area
		if (contours.size() > 0) {
			Mat area((int)rects_i.size(), 1, CV_64F);
			for (int k = 0; k < rects_i.size(); k++) { area.at<double>(k, 0) = contourArea(contours[k]); }
			int llmax;
			minMaxIdx(area, NULL, NULL, NULL, &llmax);
			RotatedRect recti = minAreaRect(contours[llmax]);
			rects_rf.push_back(recti);
		}
	}
	return rects_rf;
}

// save chosen defect into specific directory
int save_choice(Mat in, int maxloc_g, int sw) {
	// check directories
	char dp[80];
	sprintf(dp, "%1d\\", sw);
	struct stat buf;
	int u = stat(dp, &buf);
	if (u != 0) return 2;
	// save data
	char fp[80], matp[80];
	long f_cnt = 0;
	bool saved = false;
	while (!saved) {
		sprintf(fp, "%1d\\%09d.jpg", sw, f_cnt);
		struct stat buf;
		int k = stat(fp, &buf);
		if (k != 0) {
			imwrite(fp, in);
			saved = true;
			// save maxloc_g in MaxLocG.xml
			FileStorage fs;
			sprintf(matp, "%1d\\MaxLocG.xml", sw);
			struct stat bufm;
			int g = stat(matp, &bufm);
			if (g != 0) {
				fs.open(matp, FileStorage::WRITE);
				if (f_cnt >= max_num_training_data) return 1;
				Mat MaxLocG(max_num_training_data, 1, CV_8U);
				MaxLocG.setTo(cv::Scalar::all(-1));
				MaxLocG.at<uchar>(f_cnt, 0) = maxloc_g;
				fs << "MaxLocG" << MaxLocG;
				fs.release();
			}
			else {
				// read out Mat
				fs.open(matp, FileStorage::READ);
				Mat MaxLocG;
				fs["MaxLocG"] >> MaxLocG;
				fs.release();
				// write in Mat
				fs.open(matp, FileStorage::WRITE);
				if (f_cnt >= max_num_training_data) return 1;
				MaxLocG.at<uchar>(f_cnt, 0) = maxloc_g;
				fs << "MaxLocG" << MaxLocG;
				fs.release();
			}
		}
		else {
			f_cnt++;
		}
	}
	return 0;
}