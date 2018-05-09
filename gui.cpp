#include "cmvt_ai.h"

bool update_training = true, trained = false;
int nframe = 0;
int mouse_pnt_x = -1, mouse_pnt_y = -1, vLeftTop_x = -1, vLeftTop_y = -1, vRightDown_x = -1, vRightDown_y = -1;
Mat TrainingData, Classes;
// variables for AI
Ptr<TrainData> svmTrainData;
Ptr<SVM> svmClassifier = SVM::create();
Ptr<TrainData> annTrainData;
Ptr<ANN_MLP> ann = ANN_MLP::create();

// Graphical User Interface to Manually adjust Exposure-Time(us)
int GUI_ET(Mat in, Mat brightness, char *Cfont_path, int *int_exposuretime,
	double *exposuretime, double *auto_factor, bool *manual, int hist_inty_target) {
	// convert into color image for GUI only; the intrinsic image is frame_g !!
	Mat frame_g = in.clone(), frame_color;
	cvtColor(frame_g, frame_color, CV_GRAY2BGR);
	// show histogram
	int gray_max = 256, gray_min = 64, histSize = 64, maxloc_g;
	Mat img_hist = drawHist_gray(frame_g, histSize, gray_min, gray_max, &maxloc_g);
	maxloc_g = gray_min + cvRound((double)maxloc_g*(gray_max - gray_min) / (histSize - 1.0));
	imshow("Histogram", img_hist);
	// show brightness series
	Mat img_brg = drawSeries(brightness, 256);
	imshow("Brightness", img_brg);
	// automatic/manual adjust exposure time
	Rect rect1, rect2;
	if (*manual==true) {
		// generate trackbar to adjust exposure time
		createTrackbar("time(ms)", "Histogram", int_exposuretime, 200, on_trackbar_ET);
		if (*int_exposuretime < 1) *int_exposuretime = 1;
		Pylon_setET((double)(*int_exposuretime*1000.0));
		// show image
		char *str1 = "手動曝光模式(Esc鍵離開)";
		char *str2 = "切換成自動";
		rect1 = drawString(frame_color, str1, Point(10, 35), Scalar(0, 255, 0), 30.0, Cfont_path);
		rect2 = drawButton(frame_color, str2, Point(10, 40 + rect1.height), Scalar(0, 0, 255), 30.0, Cfont_path);
		imshow("Adjustment for exposure time", frame_color);
	}
	else {
		// automatic adjust
		int target_gray = hist_inty_target;
		if (target_gray > maxloc_g) *exposuretime = *exposuretime + *auto_factor;
		else { *exposuretime = *exposuretime - *auto_factor; }
		Pylon_setET(*exposuretime);
		// show image
		char *str1 = "自動曝光模式(Esc鍵離開)";
		char *str2 = "切換成手動";
		rect1 = drawString(frame_color, str1, Point(10, 35), Scalar(0, 255, 0), 30.0, Cfont_path);
		rect2 = drawButton(frame_color, str2, Point(10, 40 + rect1.height), Scalar(0, 0, 255), 30.0, Cfont_path);
		imshow("Adjustment for exposure time", frame_color);
	}
	// detect mouse-click event
	setMouseCallback("Adjustment for exposure time", on_mouse_pnt, NULL);
	if (mouse_pnt_x > rect2.x && mouse_pnt_x<(rect2.x + rect2.width) && mouse_pnt_y>rect2.y && mouse_pnt_y < (rect2.y + rect2.height)) {
		// reset parameters
		*auto_factor = 5000.0;
		mouse_pnt_x = -1;
		mouse_pnt_y = -1;
		if (*manual) { *exposuretime = (double)1000.0*(*int_exposuretime); }
		else { *int_exposuretime = cvRound(*exposuretime / 1000.0); }
		*manual = !*manual;
		destroyWindow("Histogram");
		destroyWindow("Brightness");
		destroyWindow("Adjustment for exposure time");
		waitKey(500);
	}
	// return value
	return maxloc_g;
}

// Graphical User Interface with mode [0 for detection; 1 for collection; 2 for recognition]
Mat GUI(Mat in, Mat brightness, char *Cfont_path, int *mode, int nkpnt,
	int brightness_count, double clr_diff, int kdim, int brg_range, double brg_thrsd_r,
	double rect_overlap_dist, bool ANN, bool rotatedbox, int ntype_defect, bool plt_t_contour, 
	int intpSize, int ANNlayer, int ANNneuron, int scaleL, int scaleH) {
	// convert into color image for GUI only; the intrinsic image is frame_g !!
	Mat frame_g = in.clone(), frame_color;
	cvtColor(frame_g, frame_color, CV_GRAY2BGR);
	// show brightness series
	Mat img_brg = drawSeries(brightness, 256);
	imshow("Brightness", img_brg);
	// justify defect of color difference
	double brg_avg, brg_ratio, brightness_min;
	brg_avg = sum(brightness(cv::Rect(brightness.cols - brightness_count - 1, 0, brightness_count, 1)))[0] / brightness_count;
	brg_ratio = abs(brightness.at<double>(0, brightness.cols - 1) - brg_avg) / brg_avg;
	minMaxIdx(brightness, &brightness_min, NULL, NULL, NULL);
	if (brg_ratio > clr_diff && cvRound(brightness_min) > 1) {
		char *stri = "色差";
		Rect rect1 = drawString(frame_color, stri, Point(cvRound(frame_color.cols / 2.0), cvRound(frame_color.rows / 2.0)), Scalar(0, 255, 255), 30.0, Cfont_path);
	}
	// obtain the max-accumulated gray level in histogram
	int gray_max = 256, gray_min = 64, histSize = 64, maxloc_g;
	Mat img_hist = drawHist_gray(frame_g, histSize, gray_min, gray_max, &maxloc_g);
	maxloc_g = gray_min + cvRound((double)maxloc_g*(gray_max - gray_min) / (histSize - 1.0));
	imshow("Histogram", img_hist);
	// *mode = 0
	vector<Rect> rectbox(3);                                           // for mode switcher
	vector<Rect> d_type(ntype_defect+1);                               // for defect-type selector
	if(*mode ==0){
		// finding and clustering keypoints
		vector<vector<Point>> kypcluster; // record pnt/label in k-mean method
		kypcluster = kypnt_cluster(frame_g, nkpnt, maxloc_g, kdim, brg_range);
		// convert keypoint into grouprectangles
		vector<Rect> rects = kypnt2rect(frame_g, kypcluster, maxloc_g, brg_thrsd_r, rect_overlap_dist);
		// refine rectangle by k-means method on video gray-frame
		if(rotatedbox){
			vector<RotatedRect> rects_rf = rotated_rect_refine_k(frame_g, rects, maxloc_g, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH);
			for (int i = 0; i < rects_rf.size(); i++) {                // show rect
				Point2f vertices[4];
				rects_rf[i].points(vertices);
				int cr = rand() % 255, cg = rand() % 255, cb = rand() % 255;
				for (int j = 0; j < 4; j++) line(frame_color, vertices[j], vertices[(j + 1) % 4], Scalar(cb, cg, cr), 2);
			}
		}
		else { 
			vector<Rect> rects_rf = rect_refine_k(frame_g, rects, maxloc_g, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH);
			for (int i = 0; i < rects_rf.size(); i++)                  // show rect
				rectangle(frame_color, rects_rf[i], Scalar(rand() % 255, rand() % 255, rand() % 255), 2);
		}
		// show images
		char *stri = "錄影模式(Esc鍵離開)";
		char *str2 = "切換成選取模式";
		char *str3 = "切換成ＡＩ模式";
		rectbox[0] = drawString(frame_color, stri, Point(10, 35), Scalar(0, 255, 0), 30.0, Cfont_path);
		rectbox[1] = drawButton(frame_color, str2, Point(10, 40 + rectbox[0].height), Scalar(0, 0, 255), 30.0, Cfont_path);
		rectbox[2] = drawButton(frame_color, str3, Point(10, 45 + rectbox[0].height + rectbox[1].height), Scalar(0, 0, 255), 30.0, Cfont_path);
		imshow("CMVT_AI", frame_color);
	}
	else if (*mode == 1) {
		//check if selecting some region
		if (mouse_pnt_x >= 0 && vLeftTop_x == -1) {
			vLeftTop_x = mouse_pnt_x;
			vLeftTop_y = mouse_pnt_y;
		}
		else if (mouse_pnt_x >= 0 && vLeftTop_x >= 0 && vRightDown_x == -1 && vLeftTop_x != mouse_pnt_x) {
			vRightDown_x = mouse_pnt_x;
			vRightDown_y = mouse_pnt_y;
			mouse_pnt_x = -1;
			mouse_pnt_y = -1;
		}
		char *strj = "";
		nframe = (nframe + 1) % 4;
		Scalar clr;
		if (vLeftTop_x < 0 || vRightDown_x < 0) {
			if (nframe == 0) { char *strk = "用滑鼠選取缺陷區域"; strj = strk; }
			else if (nframe == 1) { char *strk = "用滑鼠選取缺陷區域．"; strj = strk; }
			else if (nframe == 2) { char *strk = "用滑鼠選取缺陷區域．．"; strj = strk; }
			else if (nframe == 3) { char *strk = "用滑鼠選取缺陷區域．．．"; strj = strk; }
			clr = Scalar(255, 255, 0);
		}
		else {
			if (nframe == 0) { char *strk = "用滑鼠點選缺陷種類"; strj = strk; }
			else if (nframe == 1) { char *strk = "用滑鼠點選缺陷種類．"; strj = strk; }
			else if (nframe == 2) { char *strk = "用滑鼠點選缺陷種類．．"; strj = strk; }
			else if (nframe == 3) { char *strk = "用滑鼠點選缺陷種類．．．"; strj = strk; }
			clr = Scalar(0, 255, 255);
			int xi = (vLeftTop_x < vRightDown_x) ? vLeftTop_x : vRightDown_x;
			int yi = (vLeftTop_y < vRightDown_y) ? vLeftTop_y : vRightDown_y;
			int wi = abs(vLeftTop_x - vRightDown_x + 1);
			int hi = abs(vLeftTop_y - vRightDown_y + 1);
			// make sure rectange is large enough
			int wmin = 16, hmin = 16;
			if (wi < wmin) {
				int dw = cvRound((wmin - wi) / 2.0);
				if ((xi - dw) >= 0 && (xi + wi + dw) < frame_g.cols) { xi = xi - dw; }
				else if ((xi - 2 * dw) >= 0) { xi = xi - 2 * dw; }
				wi = wi + 2 * dw;
			}
			if (hi < hmin) {
				int dh = cvRound((hmin - hi) / 2.0);
				if ((yi - dh) >= 0 && (yi + hi + dh) < frame_g.rows) { yi = yi - dh; }
				else if ((yi - 2 * dh) >= 0) { yi = yi - 2 * dh; }
				hi = hi + 2 * dh;
			}
			Rect rect_chs = Rect(xi, yi, wi, hi);
			rectangle(frame_color, rect_chs, Scalar(rand() % 255, rand() % 255, rand() % 255), 2);
			Mat img_chs = frame_g(rect_chs);                           // choosen image
			// show defect-type button
			int sw = -1;
			char *del = "取消";
			char *m0 = "1：缺";
			char *m1 = "2：圓";
			char *m2 = "3：線";
			char *m3 = "4：凸";
			char *m4 = "5：泡";
			char *m5 = "6：go";
			d_type[0] = drawButton(frame_color, del, Point(frame_color.cols - 60, 35), Scalar(0, 255, 255), 30.0, Cfont_path);
			d_type[1] = drawButton(frame_color, m0, Point(frame_color.cols - 60, 40 + 1 * d_type[0].height), Scalar(0, 255, 255), 30.0, Cfont_path);
			d_type[2] = drawButton(frame_color, m1, Point(frame_color.cols - 60, 45 + 2 * d_type[0].height), Scalar(0, 255, 255), 30.0, Cfont_path);
			d_type[3] = drawButton(frame_color, m2, Point(frame_color.cols - 60, 50 + 3 * d_type[0].height), Scalar(0, 255, 255), 30.0, Cfont_path);
			d_type[4] = drawButton(frame_color, m3, Point(frame_color.cols - 60, 55 + 4 * d_type[0].height), Scalar(0, 255, 255), 30.0, Cfont_path);
			d_type[5] = drawButton(frame_color, m4, Point(frame_color.cols - 60, 60 + 5 * d_type[0].height), Scalar(0, 255, 255), 30.0, Cfont_path);
			d_type[6] = drawButton(frame_color, m5, Point(frame_color.cols - 60, 65 + 6 * d_type[0].height), Scalar(0, 255, 255), 30.0, Cfont_path);
			// check if choose the defect-type
			for (int k = 0; k <7; k++) {
				if (mouse_pnt_x > d_type[k].x && mouse_pnt_x<(d_type[k].x + d_type[k].width) && mouse_pnt_y>d_type[k].y && mouse_pnt_y < (d_type[k].y + d_type[k].height)) {
					sw = k;
					break;
				}
			}
			if (sw == 0) {                                             // clear 
				mouse_pnt_x = -1;
				mouse_pnt_y = -1;
				vLeftTop_x = -1;
				vLeftTop_y = -1;
				vRightDown_x = -1;
				vLeftTop_y = -1;
				sw = -1;
			}
			else if (sw > 0) {                                      
				int err = save_choice(img_chs, maxloc_g, sw);          // save image and data
				if (err == 1) cout << "Error on saving chosen image: too many images!" << endl;
				if (err == 2) cout << "Error on saving chosen image: directory doesn't exist!" << endl;
				if (err > 0) exit(1);
				mouse_pnt_x = -1;
				mouse_pnt_y = -1;
				vLeftTop_x = -1;
				vLeftTop_y = -1;
				vRightDown_x = -1;
				vLeftTop_y = -1;
				sw = -1;
			}
		}
		// show images
		char *stri = "選取模式(Esc鍵離開)";
		char *str2 = "切換成ＡＩ模式";
		char *str3 = "切換成錄影模式";
		rectbox[2] = drawButton(frame_color, str3, Point(10, 35), Scalar(0, 0, 255), 30.0, Cfont_path);
		rectbox[0] = drawString(frame_color, stri, Point(10, 35 + rectbox[2].height), Scalar(0, 255, 0), 30.0, Cfont_path);
		rectbox[0] = drawString(frame_color, strj, Point(10 + rectbox[0].width, 35 + rectbox[2].height), clr, 24.0, Cfont_path);
		rectbox[1] = drawButton(frame_color, str2, Point(10, 45 + rectbox[0].height + rectbox[2].height), Scalar(0, 0, 255), 30.0, Cfont_path);
		imshow("CMVT_AI", frame_color);
	}
	else if (*mode == 2) {
		// if the training data exists, do the training by ANN or ML
		if (update_training) { 
			trained = train_AI(ntype_defect, ANN, intpSize, brg_thrsd_r, rect_overlap_dist, 1, 1, plt_t_contour);
			if (trained) {
				// load AI data
				FileStorage fs_ai;
				if (ANN) { fs_ai.open("ANN.xml", FileStorage::READ); }
				else { fs_ai.open("ML.xml", FileStorage::READ); }
				fs_ai["TrainingData"] >> TrainingData;
				fs_ai["Classes"] >> Classes;
				fs_ai.release();
				// training data for AI
				if (ANN) {
					annTrainData = TrainData::create(TrainingData, ROW_SAMPLE, Classes);                     // group training data
					Mat_<int> layerSizes(1, ANNlayer+2);                                                     // set parameters
					layerSizes(0, 0) = TrainingData.cols;
					for (int k = 1; k <= ANNlayer; k++) layerSizes(0, k) = ANNneuron;
					layerSizes(0, ANNlayer + 1) = ntype_defect;
					ann->setLayerSizes(layerSizes);
					ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
					ann->setTrainMethod(ANN_MLP::BACKPROP, 0.0001, 0.0001);
					ann->train(annTrainData);                                                                // proceed training
				}
				else {
					svmTrainData = TrainData::create(TrainingData, ROW_SAMPLE, Classes);                     // group training data
					svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 0.01));        // set parameters
					svmClassifier->setC(0.1);
					svmClassifier->setKernel(SVM::LINEAR);
					svmClassifier->train(svmTrainData);                                                      // proceeding training
				}
			}
		}
		update_training = false;
		if (!trained) {
			char *str_msg = "訓練資料不存在";
			Rect rect_msg = drawString(frame_color, str_msg, Point(cvRound(frame_color.cols / 2.0) - 100, frame_color.rows - 35), Scalar(0, 255, 255), 30.0, Cfont_path);
		}
		else {
			if (ANN) {
				char *str_msg = "AI模式=ANN";
				Rect rect_msg = drawString(frame_color, str_msg, Point(cvRound(frame_color.cols / 2.0) - 30, frame_color.rows - 35), Scalar(0, 255, 255), 30.0, Cfont_path);
			}
			else {
				char *str_msg = "AI模式=SVM";
				Rect rect_msg = drawString(frame_color, str_msg, Point(cvRound(frame_color.cols / 2.0) - 30, frame_color.rows - 35), Scalar(0, 255, 255), 30.0, Cfont_path);
			}
			// finding and clustering keypoints for real-time image
			vector<vector<Point>> kypcluster;                          // record pnt/label in k-mean method
			kypcluster = kypnt_cluster(frame_g, nkpnt, maxloc_g, kdim, brg_range);
			// convert keypoint into grouprectangles
			vector<Rect> rects = kypnt2rect(frame_g, kypcluster, maxloc_g, brg_thrsd_r, rect_overlap_dist);
			// refine rectangle by k-means method on video gray-frame
			vector<Rect> rects_rf = rect_refine_k(frame_g, rects, maxloc_g, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH);
			for (int i = 0; i < rects_rf.size(); i++) {
				// calculate feature
				Mat featurei = featurem(frame_g, rects_rf[i], maxloc_g, intpSize, brg_thrsd_r, rect_overlap_dist, scaleL, scaleH, 0, ntype_defect);
				int response, opp;
				if (ANN) {
					Mat p(1, 2 + intpSize * intpSize, CV_32F);;
					featurei(cv::Rect(0, 0, 2 + intpSize * intpSize, 1)).copyTo(p(cv::Rect(0, 0, 2 + intpSize * intpSize, 1)));
					Mat output;
					ann->predict(p, output);
					Point maxLoc;
					double maxVal;
					minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
					response = maxLoc.x + 1;
					opp = cvRound(100.0*output.at<float>(0, maxLoc.x) / sum(output)[0]);
				}
				else {
					Mat p(1, 2 + intpSize * intpSize, CV_32F);;
					featurei(cv::Rect(0, 0, 2 + intpSize * intpSize, 1)).copyTo(p(cv::Rect(0, 0, 2 + intpSize * intpSize, 1)));
					response = (int)svmClassifier->predict(p);
				}
				// plot upright bounding box
				if (response >= 1 && response <= 5) {
					int cr = rand() % 255, cg = rand() % 255, cb = rand() % 255;
					rectangle(frame_color, rects_rf[i], Scalar(cb, cg, cr), 2);
					String str_tmp;
					if (ANN) {
						if (response == 1) str_tmp = String("缺") + std::to_string(opp) + String("%");
						if (response == 2) str_tmp = String("圓") + std::to_string(opp) + String("%");
						if (response == 3) str_tmp = String("線") + std::to_string(opp) + String("%");
						if (response == 4) str_tmp = String("凸") + std::to_string(opp) + String("%");
						if (response == 5) str_tmp = String("泡") + std::to_string(opp) + String("%");
					}
					else {
						if (response == 1) str_tmp = String("缺");
						if (response == 2) str_tmp = String("圓");
						if (response == 3) str_tmp = String("線");
						if (response == 4) str_tmp = String("凸");
						if (response == 5) str_tmp = String("泡");
					}
					char *str_prd = new char[str_tmp.length() + 1];
					strcpy(str_prd, str_tmp.c_str());
					int ypos;
					if ((rects_rf[i].y - 15) > 15) { ypos = rects_rf[i].y - 15; }
					else { ypos = rects_rf[i].y + rects_rf[i].height + 15; }
					Rect rect_prd = drawString(frame_color, str_prd, Point(rects_rf[i].x, ypos), Scalar(cb, cg, cr), 30.0, Cfont_path);
				}
			}
		}
		// show images
		char *stri = "ＡＩ模式(Esc鍵離開)";
		char *str2 = "切換成錄影模式";
		char *str3 = "切換成選取模式";
		rectbox[1] = drawButton(frame_color, str2, Point(10, 35), Scalar(0, 0, 255), 30.0, Cfont_path);
		rectbox[2] = drawButton(frame_color, str3, Point(10, 40 + rectbox[1].height), Scalar(0, 0, 255), 30.0, Cfont_path);
		rectbox[0] = drawString(frame_color, stri, Point(10, 40 + rectbox[1].height + rectbox[2].height), Scalar(0, 255, 0), 30.0, Cfont_path);
		imshow("CMVT_AI", frame_color);
	}
	// detect mouse-click event
	setMouseCallback("CMVT_AI", on_mouse_pnt, NULL);
	for (int k = 1; k <= 2; k++) {
		if (mouse_pnt_x > rectbox[k].x && mouse_pnt_x<(rectbox[k].x + rectbox[k].width) && mouse_pnt_y>rectbox[k].y && mouse_pnt_y < (rectbox[k].y + rectbox[k].height)) {
			// reset parameters
			mouse_pnt_x = -1;
			mouse_pnt_y = -1;
			vLeftTop_x = -1;
			vLeftTop_y = -1;
			vRightDown_x = -1;
			vLeftTop_y = -1;
			*mode = (*mode + k) % 3;
			destroyWindow("Histogram");
			destroyWindow("CMVT_AI");
			destroyWindow("Brightness");
			update_training = true;
			trained = false;
			TrainingData.release();
			Classes.release();
			waitKey(500);
			break;
		}
	}
	return frame_color;
}
