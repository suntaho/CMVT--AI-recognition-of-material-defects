#include "cmvt_ai.h"

extern long max_num_training_data;

// quantify features of assigned patch in gray-image, as input of ai-training/ai-recognition
Mat featurem(Mat in, Rect rect_in, int maxloc_g, int intpSize, double brg_thrsd_r, 
	double rect_overlap_dist, int scaleL, int scaleH, int typei, int ntype_defect) {
	Mat img = in.clone();
	int data_size = 2 + intpSize * intpSize + 4;                              // the last 4 elements record rectangles
	Mat feature = Mat::zeros(1, data_size, CV_32FC1);
	// maxloc_g
	feature.at<float>(0, 0) = (float)maxloc_g;
	// calculate width/height or height/width (reuqired by >1) by rotated rectangle method
	vector<Rect> range;
	range.push_back(rect_in);
	vector<vector<Point>> contours = cal_closed_contours_maxArae(img, range, maxloc_g, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH);
	vector<RotatedRect> rotated_recti;
	vector<Rect> recti;
	if (contours.size() > 0) {
		RotatedRect rotated_rc = minAreaRect(contours[0]);
		rotated_recti.push_back(rotated_rc);
		Rect rc = boundingRect(contours[0]);
		recti.push_back(rc);
	}
	if (rotated_recti.size() > 0) {
		double ratio_wh;
		ratio_wh = rotated_recti[0].size.width / rotated_recti[0].size.height;
		if (ratio_wh < 1.0) ratio_wh = 1.0 / ratio_wh;
		feature.at<float>(0, 1) = (float)ratio_wh;
	}
	else {
		feature.at<float>(0, 1) = (float) 1.0;
	}
	if (recti.size() > 0 && typei < ntype_defect) {
		// additional data
		feature.at<float>(0, 2 + intpSize * intpSize) = (float)recti[0].x;
		feature.at<float>(0, 2 + intpSize * intpSize + 1) = (float)recti[0].y;
		feature.at<float>(0, 2 + intpSize * intpSize + 2) = (float)recti[0].width;
		feature.at<float>(0, 2 + intpSize * intpSize + 3) = (float)recti[0].height;
	}
	else {
		// additional data
		feature.at<float>(0, 2 + intpSize * intpSize) = (float)rect_in.x;
		feature.at<float>(0, 2 + intpSize * intpSize + 1) = (float)rect_in.y;
		feature.at<float>(0, 2 + intpSize * intpSize + 2) = (float)rect_in.width;
		feature.at<float>(0, 2 + intpSize * intpSize + 3) = (float)rect_in.height;
	}
	// calculate accumulation histogram along x and y on the resized image
	Mat img_cap;
	if (recti.size() > 0 && typei < ntype_defect) { img_cap = img(recti[0]).clone(); }
	else { img_cap = img(rect_in).clone(); }
	if (rect_in.width > 2 && rect_in.height > 2) {
		Mat rimg(intpSize, intpSize, CV_8UC1);
		resize(img_cap, rimg, rimg.size(), 0, 0, INTER_LINEAR);
		// normalize rimg
		for (int k = 0; k < rimg.rows; k++) {
			uchar* inData = rimg.ptr<uchar>(k);
			for (int i = 0; i < rimg.cols; i++)
				inData[i] = cvRound(abs(inData[i] - maxloc_g));
		}
		// calculate pixel accumulation along x and then y
		for (int i = 0; i < intpSize; i++) {
			uchar* inData = rimg.ptr<uchar>(i);
			for (int j = 0; j < intpSize; j++) {
				int k = i*intpSize + j;
				feature.at<float>(0, 2 + k) = (float)1.0*inData[j] / maxloc_g;
			}
		}
	}
	return feature;
}

// training data, if data exists
bool train_AI(int ntype_defect, bool ANN, int intpSize, double brg_thrsd_r, double rect_overlap_dist,
	int scaleL, int scaleH, bool plt_t_contour) {
	bool trained = false;
	Mat TrainingData;
	Mat Classes;
	// read images and xml for traing
	Mat feature_all(ntype_defect*max_num_training_data, 2 + intpSize * intpSize + 4, CV_32FC1);
	vector<float> class_all;
	long f_cnt_trained = 0;
	for (int sw = 1; sw <= ntype_defect; sw++) {
		// check directories
		char dp[80];
		sprintf(dp, "%1d\\", sw);
		struct stat buf;
		int u = stat(dp, &buf);
		if (u != 0) { continue; }
		// check xml
		FileStorage fs_xml;
		Mat MaxLocG;
		sprintf(dp, "%1d\\MaxLocG.xml", sw);
		u = stat(dp, &buf);
		if (u != 0) { continue; }
		else {
			fs_xml.open(dp, FileStorage::READ);
			fs_xml["MaxLocG"] >> MaxLocG;
			fs_xml.release();
		}
		// load image and proceed training
		char fp[80], fpo[80];
		long f_cnt = 0;
		while (f_cnt < max_num_training_data) {
			sprintf(fp, "%1d\\%09d.jpg", sw, f_cnt);
			sprintf(fpo, "%1d\\c_%09d.jpg", sw, f_cnt);
			struct stat buf;
			int k = stat(fp, &buf);
			if (k == 0) {                                              // file exist. do training
				Mat imgi = imread(fp, CV_LOAD_IMAGE_GRAYSCALE);
				int maxloc_g = MaxLocG.at<uchar>(f_cnt, 0);
				Rect recti(0, 0, imgi.cols, imgi.rows);
				Mat featurei = featurem(imgi, recti, maxloc_g, intpSize, brg_thrsd_r,
					rect_overlap_dist, scaleL, scaleH, sw, ntype_defect);
				if (featurei.rows > 0) {
					float* Data1 = feature_all.ptr<float>(f_cnt_trained);
					float* Data2 = featurei.ptr<float>(0);
					for (int g = 0; g < featurei.cols; g++) { Data1[g] = Data2[g]; }
					class_all.push_back((float)1.0*sw);
					f_cnt_trained++;
					// OPTIONAL: convert jpg to another contoured jpg for benchmark;
					if (plt_t_contour) {
						int x, y, width, height;
						x = (int)featurei.at<float>(0, 2 + intpSize * intpSize);
						y = (int)featurei.at<float>(0, 2 + intpSize * intpSize + 1);
						width = (int)featurei.at<float>(0, 2 + intpSize * intpSize + 2);
						height = (int)featurei.at<float>(0, 2 + intpSize * intpSize + 3);
						Rect rectp(x, y, width, height);
						Mat imgc;
						cvtColor(imgi, imgc, CV_GRAY2BGR);
						rectangle(imgc, rectp, Scalar(rand() % 255, rand() % 255, rand() % 255), 2);
						imwrite(fpo, imgc);
					}
				}
			}
			f_cnt++;
		}
	}
	// build TrainingData and Classes
	if (f_cnt_trained > 0) {
		long nsample = f_cnt_trained, nvars = 2 + intpSize * intpSize;
		TrainingData.create(nsample, nvars, CV_32FC1);
		if (ANN) { Classes.create(nsample, ntype_defect, CV_32FC1); }
		else { Classes.create(nsample, 1, CV_32S); }
		Classes.setTo(cv::Scalar::all(0));
		// copy TrainingData
		feature_all(cv::Rect(0, 0, nvars, nsample)).copyTo(TrainingData(cv::Rect(0, 0, nvars, nsample)));
		// copy Classes data
		for (int k = 0; k < nsample; k++) {
			int typei = cvRound(class_all[k]);
			if (ANN) { Classes.at<float>(k, typei - 1) = (float) 1.0*typei; }
			else { Classes.at<int>(k, 0) = typei; }
		}
	}
	// save AI data
	if (f_cnt_trained > 0) {
		trained = true;
		cout << TrainingData.rows << " samples are trained!" << endl;
		// save AI data
		FileStorage fs_ai;
		if (ANN) { fs_ai.open("ANN.xml", FileStorage::WRITE); }
		else { fs_ai.open("ML.xml", FileStorage::WRITE); }
		fs_ai << "TrainingData" << TrainingData;
		fs_ai << "Classes" << Classes;
		fs_ai.release();
	}
	return trained;
}
