#define _CRT_SECURE_NO_WARNINGS

#include "DetectController.h"
#include <algorithm>
#include <io.h>
#include <Windows.h>

void readDirectory(const string& directoryName, vector<string>& filenames, const string& suffix);

void shufflePixel(Mat &X, Mat &Y);

void cameraSkin();
void cameraGesture();
void trainSkinDetectorOld();


void takePhoto();
void detectEdge();
void selectSingleThresh();
void selectHysterThresh();
void crossValidate();
double VFCV_singleThresh(Mat allSetX, Mat allSetY, double thresh, int v = 5);
double VFCV_hysterThresh(HandFaceDetect &detect, Mat allSetX, Mat allSetY, double threshMax, double threshMin, int v = 5);
pair<Mat, Mat> readAllSet();
pair<Mat, Mat> readFullPicture();

void learningCurve();
void filter();
void evaluateHyster();
void trainSkinDetector();
void extractForeground();

void transfer() {
	// 转移图片
	vector<string> picNames;
	readDirectory("original_images", picNames, "jpg");
	int num = picNames.size();
	
	for (unsigned i = 0; i < 62; i++) {
		char originName[50] = "";
		char skinName[50] = "";
		sprintf(originName, "Seed Images\\origin%d.jpg", 1 + i);
		sprintf(skinName, "Seed Images\\skin_masks\\skin%d.jpg", 1 + i);
		Mat origin = imread(originName);
		Mat skin = imread(skinName);

		sprintf(originName, "original_images\\origin%04d.jpg", num);
		sprintf(skinName, "skin_masks\\mask%04d.jpg", num);
		++num;

		imwrite(originName, origin);
		imwrite(skinName, skin);
	}
}

int main(){
//	trainSkinDetectorOld();
//	extractForeground();
//	cameraSkin();

//	cameraGesture();
	takePhoto();
//	test();
//	detectEdge();

//	crossValidate();
//	vFoldCrossValidate();
//	selectHysterThresh();
	/*
	DetectController ctrl;	
	Mat testImage = imread("Test Images\\test.jpg");
	imshow("testImage", testImage);
	ctrl.detectHandFace(testImage);
	imshow("result image", ctrl.getResultImage());
	*/

//	waitKey(0);
//	system("PAUSE ");
	return 0;
}


void detectEdge() {
	cv::VideoCapture capture(0);
	// check if video successfully opened
	if (!capture.isOpened()) {
		cout << "camera open fail" << endl;
		return;
	}
	Mat frame; // current video frame

	while (waitKey(30) != 27) {



		// read next frame if any
		if (!capture.read(frame))
			break;

		imshow("Extracted Frame", frame);

		Mat edges;
		Canny(frame, edges, 90, 350); // 125, 350
		imshow("edges", edges);
	}
	// Close the video file.
	// Not required since called by destructor
	capture.release();
}

void takePhoto() {
	//- 照相并保存, 按空格保存，按Esc退出

	cv::VideoCapture capture(0);
	// check if video successfully opened
	if (!capture.isOpened()) {
		cout << "camera open fail" << endl;
		return;
	}
	Mat frame; // current video frame

	vector<string> picNames;
	readDirectory("original_images", picNames, "jpg");
	int picNum = picNames.size();
	while (1) {
		// read next frame if any
		if (!capture.read(frame))
			break;

		imshow("Extracted Frame", frame);

		char c = waitKey(30);
		//if(c!=-1) cout << (int)c << endl;
		if (c == 27) break;
		else if(c==32) {
			printf("take a photo\n");
			char pic_name[50] = "";
			sprintf(pic_name, "original_images\\origin%04d.jpg", picNum++);
			imwrite(pic_name,frame);
		}
	}
	// Close the video file.
	// Not required since called by destructor
	capture.release();
}


void readDirectory(const string& directoryName, vector<string>& filenames, const string& suffix)
{
	// 读取路径下的所有文件名
	filenames.clear();
	struct _finddata_t s_file;
	string str = directoryName + "\\*." +suffix;

	intptr_t h_file = _findfirst(str.c_str(), &s_file);
	if (h_file != static_cast<intptr_t>(-1.0)) {
		do {
			filenames.push_back(directoryName + "\\" + s_file.name);
		} while (_findnext(h_file, &s_file) == 0);
	}
	_findclose(h_file);

	sort(filenames.begin(), filenames.end());
}

void cameraSkin() {
	// Open the video file
	cv::VideoCapture capture(0);
	// check if video successfully opened
	if (!capture.isOpened()) {
		cout << "camera open fail" << endl;
		return;
	}
	Mat frame; // current video frame

	SkinClassifier skinClassifier;
	skinClassifier.loadOld("Seed Results");


	while (waitKey(30) != 27) {
		// read next frame if any
		if (!capture.read(frame))
			break;

		resize(frame, frame, Size(320, 240));
		imshow("Extracted Frame", frame);

		Mat probImg = skinClassifier.predictProbImg(frame);
		imshow("probImg", probImg);
		imshow("color", skinClassifier.cvtGrayToColor(probImg));
		Mat afterThresh;
		threshold(probImg, afterThresh, 255*0.2, 255, THRESH_BINARY);
		imshow("afterThresh", afterThresh);
	}
	// Close the video file.
	// Not required since called by destructor
	capture.release();
}

void cameraGesture(){
	// Open the video file
	cv::VideoCapture capture(0);
	// check if video successfully opened
	if (!capture.isOpened()) {
		cout << "camera open fail" << endl;
		return;
	}

	Mat frame; // current video frame
 
	DetectController controller;
	static int count = 0;
	while (waitKey(30) != 27) {
		// read next frame if any
		
		if (!capture.read(frame))
			break;

		resize(frame, frame, Size(320, 240));
		imshow("原图", frame);
		auto pre = GetTickCount();
		controller.detectHandFace(frame);
		cout << GetTickCount() - pre << endl;
		imshow("最终结果", controller.getResultImage());
		imwrite("result_image\\" + to_string(count) + ".jpg", controller.getResultImage());
		count = (count + 1) % 50;
	}
	// Close the video file.
	// Not required since called by destructor
	capture.release();
}

void trainSkinDetectorOld() {
	//- 训练肤色分类器

	vector<string> origin_names, mask_names;
	readDirectory("original_images", origin_names, "jpg");
	readDirectory("skin_masks", mask_names, "jpg");
	auto name_len = origin_names[0].size();
	Mat allSetX, allSetY;
	for (unsigned i = 0; i < origin_names.size(); i++) {
		auto origin_index = origin_names[i].substr(origin_names[0].size() - 8);
		auto mask_index = mask_names[i].substr(mask_names[0].size() - 8);
		if (origin_index != mask_index)
			throw exception("name don't match"); // 检查原图与皮肤图的编号是否匹配

		Mat origin = imread(origin_names[i]);
		allSetX.push_back(origin.reshape(origin.channels(), origin.total())); // 将图像尺寸转换成pixelNum*1再连成一长列
		Mat mask = imread(mask_names[i], 0);
		allSetY.push_back(mask.reshape(mask.channels(), mask.total()));
	}
	threshold(allSetY, allSetY, 1, 255, THRESH_BINARY);

	SkinClassifier skin;
	skin.setSkinImg(allSetX);
	skin.setSkinMask(allSetY);
	skin.train();
	skin.saveOld("Seed Results"); // 整个项目只有此处会写probTable.xml文件
	printf("train old done\n");
}

void trainSkinDetector() {
	SkinClassifier skin;

	printf("Loading nonSkinImgs....");
	vector<string> names;
	readDirectory("Seed Images", names, "jpg");
	for (auto name : names) {
		Mat nonSkinImg = imread(name);
		skin.setNonSkinImg(nonSkinImg);
		skin.addNonSkinHist();
	}
	printf("Done.\nLoading skinImgs....");

	vector<string> imgNames, maskNames;
	readDirectory("F:\\手势识别\\图库\\HGR database\\original_images", imgNames, "jpg");
	readDirectory("F:\\手势识别\\图库\\HGR database\\skin_masks", maskNames, "bmp");
	int num = imgNames.size();
	for (int i = 0; i < num;i++) {
		Mat skinImg = imread(imgNames[i]);
		skin.setSkinImg(skinImg);
		Mat skinMask = imread(maskNames[i], 0);
		threshold(skinMask, skinMask, 1, 255, THRESH_BINARY_INV);
		skin.setSkinMask(skinMask);
		skin.addSkinHist();
	}

	printf("Done.\nTraining....");
	skin.trainSN();
	skin.save("Seed Results");
	printf("Done.\n");
}

void shufflePixel(Mat &X, Mat &Y) {
	//-- 随机化图片像素
	// X: 彩色原始图片
	// Y: 灰色皮肤图片
	// X和Y的size要相同
	const int random_seed = 4;
	X = X.reshape(1, X.total()); // size: picNum*3
	Y = Y.reshape(1, Y.total()); // size: picNum*1
	X = X.t();
	Y = Y.t();
	X.push_back(Y);
	X = X.t();  // size: picNum*4
	X = X.reshape(4);
	randShuffle(X, random_seed);
	X = X.reshape(1);
	Y = X.colRange(3, 4); // size: picNum*1
	X = X.colRange(0, 3);
	X = X.reshape(3); // size: picNum*1

//	imshow("Y", Y.clone().reshape(3, 62 * 240));
}

void crossValidate() {

	const double trainingSetPercent = 0.7;
	const double cvSetPercent = 0.3;
	printf("trainingSet: %d%%,  cvSet: %d%% \n",
		static_cast<int>(trainingSetPercent * 100), static_cast<int>(cvSetPercent * 100));
	
	// 读取样本
	auto allSet = readAllSet();
	auto allSetX = allSet.first;
	auto allSetY = allSet.second;

	int pixelSum = allSetX.rows;

	Mat trainingSetX, trainingSetY;
	trainingSetX = allSetX.rowRange(0, static_cast<int>(trainingSetPercent * pixelSum));
	trainingSetY = allSetY.rowRange(0, static_cast<int>(trainingSetPercent * pixelSum));

	Mat cvSetX, cvSetY;
	cvSetX = allSetX.rowRange(static_cast<int>(trainingSetPercent * pixelSum), pixelSum);
	cvSetY = allSetY.rowRange(static_cast<int>(trainingSetPercent * pixelSum), pixelSum);

	SkinClassifier skinClassifier;
	skinClassifier.setSkinImg(trainingSetX);
	skinClassifier.setSkinMask(trainingSetY);
	skinClassifier.train();    // 训练

	cout << "trainingSetError:     cvSetError:\n\n";

	

	// 预测trainingSet
	skinClassifier.predictAndEvaluate(trainingSetX, trainingSetY, 0.5);

	// 预测cvSet
	skinClassifier.predictAndEvaluate(cvSetX, cvSetY, 0.5);

	cout << endl;
}

double VFCV_singleThresh(Mat allSetX, Mat allSetY, double thresh, int v) {


	int pixelSum = allSetX.rows;
	int foldPixelSum = pixelSum / v;
	allSetX.push_back(allSetX.clone()); //复制一遍，方便划分
	allSetY.push_back(allSetY.clone());

	double error = 0;
	for (int i = 0; i < v; i++) {
		Mat trainingSetX, trainingSetY;
		trainingSetX = allSetX.rowRange(foldPixelSum*(i + 1), foldPixelSum*(i + v));
		trainingSetY = allSetY.rowRange(foldPixelSum*(i + 1), foldPixelSum*(i + v));

		Mat cvSetX, cvSetY;
		cvSetX = allSetX.rowRange(foldPixelSum*i, foldPixelSum*(i+1));
		cvSetY = allSetY.rowRange(foldPixelSum*i, foldPixelSum*(i + 1));

		SkinClassifier skinClassifier;
		skinClassifier.setSkinImg(trainingSetX);
		skinClassifier.setSkinMask(trainingSetY);
		skinClassifier.train(); // 训练

		double e = skinClassifier.predictAndEvaluate(cvSetX, cvSetY, thresh); // 预测
		error += e;
	}
	error /= v;
	
	return error;
}

double VFCV_hysterThresh(HandFaceDetect &detect, Mat allSetX, Mat allSetY, double threshMax, double threshMin, int v) {


	int rowNum = allSetX.rows;
	int foldRowNum = rowNum / v;
	allSetX.push_back(allSetX.clone()); //复制一遍，方便划分
	allSetY.push_back(allSetY.clone());

	double error = 0;
	for (int i = 0; i < v; i++) {
		Mat trainingSetX, trainingSetY;
		trainingSetX = allSetX.rowRange(foldRowNum*(i + 1), foldRowNum*(i + v));
		trainingSetY = allSetY.rowRange(foldRowNum*(i + 1), foldRowNum*(i + v));

		Mat cvSetX, cvSetY;
		cvSetX = allSetX.rowRange(foldRowNum*i, foldRowNum*(i + 1));
		cvSetY = allSetY.rowRange(foldRowNum*i, foldRowNum*(i + 1));

		SkinClassifier skinClassifier;
		skinClassifier.setSkinImg(trainingSetX);
		skinClassifier.setSkinMask(trainingSetY);
		skinClassifier.train(); // 训练

		detect.setHist(skinClassifier.getProbTable());

		double e = detect.predictAndEvaluate(cvSetX, cvSetY, threshMax, threshMin); // 预测
		error += e;
	}
	error /= v;

	return error;
}

void evaluateHyster() {
	//////  找最佳的Hysterthresh阈值  //////

	const int picNum = 62;

	SkinClassifier skin;
	HandFaceDetect detect;
	skin.load("Seed Results\\probTable.xml");
	detect.load("Seed Results\\probTable.xml");

	Mat allSetX, allSetY;
	char originName[50] = "";
	char skinName[50] = "";
	for (unsigned i = 0; i < picNum; i++) {
		sprintf(originName, "Seed Images\\origin%d.jpg", 1 + i);
		sprintf(skinName, "Seed Images\\skin%d.jpg", 1 + i);
		allSetX.push_back(imread(originName));
		allSetY.push_back(imread(skinName, 0));
	}
	threshold(allSetY, allSetY, 1, 255, THRESH_BINARY);

	skin.setSkinImg(allSetX);
	skin.setSkinMask(allSetY);
	skin.train();
	detect.setHist(skin.getProbTable());

	skin.predictAndEvaluate(allSetX, allSetY,0.5);

	cout << "after hyster thresh:" << endl;
	double maxF1Score = 0.0;
	double optimumMax, optimumMin;
	for (double max = 0.8; max < 1.0; max += 0.01) {
		for (double min = 0.3; min < 0.5; min += 0.01) {
			printf("max: %.2f  min: %.2f    ", max, min);
			double f1Score = detect.predictAndEvaluate(allSetX, allSetY, max, min);
			if (f1Score > maxF1Score) {
				maxF1Score = f1Score;
				optimumMax = max;
				optimumMin = min;
			}
		}
	}

	cout << "End" << endl;
}

void learningCurve(){
	//- 绘制肤色分类器的学习曲线

	///////////  设置参数  //////////////
	const int width = 1200;
	const int height = 600;
	const int yScale = 10;    //  纵坐标放大倍数

	const int picNum = 62;
	const double trainingSetMax = 0.7;
	const double cvSetPercent = 0.3;

	const int random_seed = 3;
	printf("picNum: %d,  trainingSetMax: %d%%,  cvSetPercent: %d%%,  randomSeed: %d\n",
		picNum, static_cast<int>(trainingSetMax * 100), static_cast<int>(cvSetPercent * 100), random_seed);

	// 装载图片并分配cvSet
	Mat allSetX, allSetY;
	char originName[50] = "";
	char skinName[50] = "";
	for (unsigned i = 0; i < picNum; i++) {
		sprintf(originName, "Seed Images\\origin%d.jpg", 1 + i);
		sprintf(skinName, "Seed Images\\skin%d.jpg", 1 + i);
		allSetX.push_back(imread(originName));
		allSetY.push_back(imread(skinName, 0));
	}
	threshold(allSetY, allSetY, 1, 255, THRESH_BINARY);

	// 随机化像素顺序
	allSetX = allSetX.reshape(1, allSetX.total());
	allSetY = allSetY.reshape(1, allSetY.total());
	allSetX = allSetX.t();
	allSetY = allSetY.t();
	allSetX.push_back(allSetY);
	allSetX = allSetX.t();
	allSetX = allSetX.reshape(4);
	randShuffle(allSetX, random_seed);
	allSetX = allSetX.reshape(1);
	allSetY = allSetX.colRange(3, 4);
	allSetX = allSetX.colRange(0, 3);
	allSetX = allSetX.reshape(3);

	int pixelSum = allSetX.rows;

	Mat cvSetX, cvSetY;
	cvSetX = allSetX.rowRange(static_cast<int>(trainingSetMax * pixelSum), pixelSum);
	cvSetY = allSetY.rowRange(static_cast<int>(trainingSetMax * pixelSum), pixelSum);

	Mat learingCurve(height, width, CV_8UC3, Scalar::all(255));
	putText(learingCurve, "trainingSetError", Point(width - 120, 40), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
	putText(learingCurve, "cvSetError", Point(width - 120, 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0));
	putText(learingCurve, "10%", Point(0, 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar::all(0));
	putText(learingCurve, "5%", Point(0, height / 2), CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar::all(0));
	putText(learingCurve, "0%", Point(0, height - 2), CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar::all(0));
	char chX[10] = "";

	vector<Point> trainingErrorPoints;
	vector<Point> cvErrorPoints;

	///////////////  装载越来越多的训练集并训练和预测  //////////////////
	cout << "trainingSetError:     cvSetError:\n\n";
	for (double trainingSetPercent = 0.05; trainingSetPercent < 1.01; trainingSetPercent += 0.05) {

		sprintf(chX, "%.2f", trainingSetPercent);
		putText(learingCurve, chX, Point(static_cast<int>(width * trainingSetPercent * 0.99 - 10), height - 2), CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar::all(0));
		printf("trainingSetPercent: %d%%    \t", static_cast<int>(trainingSetPercent * 100));

		// 训练trainingSet
		SkinClassifier skinClassifier;

		Mat trainingSetX, trainingSetY;
		trainingSetX = allSetX.rowRange(0, static_cast<int>(trainingSetMax * trainingSetPercent * pixelSum));
		trainingSetY = allSetY.rowRange(0, static_cast<int>(trainingSetMax * trainingSetPercent * pixelSum));

		skinClassifier.setSkinImg(trainingSetX);
		skinClassifier.setSkinMask(trainingSetY);
		skinClassifier.train();    //  训练

		// 预测trainingSet		
		double trainingSetError = skinClassifier.predictAndEvaluate(trainingSetX, trainingSetY,0.5);
		Point trainP(static_cast<int>(width * trainingSetPercent * 0.99), static_cast<int>(height * (1 - trainingSetError * yScale)));
		cv::circle(learingCurve, trainP, 2, Scalar(0, 0, 255), -1);
		trainingErrorPoints.push_back(trainP);

		// 预测cvSet
		double cvSetError = skinClassifier.predictAndEvaluate(cvSetX, cvSetY,0.5);
		Point cvP(static_cast<int>(width * trainingSetPercent * 0.99), static_cast<int>(height * (1 - cvSetError * yScale)));
		cv::circle(learingCurve, cvP, 2, Scalar(255, 0, 0), -1);
		cvErrorPoints.push_back(cvP);

		cout << "\n\n";
	}

	//绘制learning curve
	for (unsigned i = 0; i < trainingErrorPoints.size() - 1; i++) {
		line(learingCurve, trainingErrorPoints[i], trainingErrorPoints[i + 1], Scalar(0, 0, 255));
		line(learingCurve, cvErrorPoints[i], cvErrorPoints[i + 1], Scalar(255, 0, 0));
	}
	cv::imshow("learning curve", learingCurve);
}

void selectSingleThresh() {
	// 读取样本
	auto allSet = readAllSet();
	auto allSetX = allSet.first;
	auto allSetY = allSet.second;

	double maxF1 = 0;
	double bestThresh = 0;
	for (int i = 1; i < 10; i++) {
		double thresh = 0.3+0.02*i;
		double f1 = VFCV_singleThresh(allSetX, allSetY, thresh);
		if (f1 > maxF1) {
			maxF1 = f1;
			bestThresh = thresh;
		}
		printf("thresh: %.2f, F1Score: %.4f, best thresh: %.2f\n", thresh, f1, bestThresh);
	}
	printf("best thresh: %.2f, F1Score: %.4f\n", bestThresh, maxF1);
}

void selectHysterThresh() {
	FILE *out;
	out = fopen("Seed Results\\selectHysterThresh.txt", "w");
	// 读取样本
	auto allSet = readFullPicture();
	auto allSetX = allSet.first;
	auto allSetY = allSet.second;

	// 单阈值结果，作为baseline
	double f1_single = VFCV_singleThresh(allSetX, allSetY, 0.4);
	printf("single thresh: thresh: %.2f, F1Score: %.4f\n", 0.4, f1_single);
	fprintf(out, "single thresh: thresh: %.2f, F1Score: %.4f\n", 0.4, f1_single);

	double maxF1 = 0;
	double bestThreshMax = 0;
	double bestThreshMin = 0;
	HandFaceDetect detect;
	for (double max = 0.9; max < 1.0; max += 0.02) {
		for (double min = 0.2; min < max; min += 0.02) {
			double f1 = VFCV_hysterThresh(detect, allSetX, allSetY, max, min);
			if (f1 > maxF1) {
				maxF1 = f1;
				bestThreshMax = max;
				bestThreshMin = min;
			}
			printf("Max: %.2f, Min: %.2f, F1Score: %.4f, best Max: %.2f, Min: %.2f\n", max, min, f1, bestThreshMax, bestThreshMin);
			fprintf(out, "Max: %.2f, Min: %.2f, F1Score: %.4f, best Max: %.2f, Min: %.2f\n", max, min, f1, bestThreshMax, bestThreshMin);
		}
	}
	printf("best threshMax: %.2f, threshMin: %.2f, F1Score: %.4f\n", bestThreshMax, bestThreshMin, maxF1);
	fprintf(out, "best threshMax: %.2f, threshMin: %.2f, F1Score: %.4f\n", bestThreshMax, bestThreshMin, maxF1);
	fclose(out);
}

pair<Mat, Mat> readAllSet() {
	// 读取所有样本，排成一列像素，并随机化
	vector<string> origin_names, mask_names;
	readDirectory("original_images", origin_names, "jpg");
	readDirectory("skin_masks", mask_names, "jpg");
	printf("picNum: %d\n", origin_names.size());

	Mat allSetX, allSetY;
	for (unsigned i = 0; i < origin_names.size(); i++) {
		auto origin_index = origin_names[i].substr(origin_names[0].size() - 8);
		auto mask_index = mask_names[i].substr(mask_names[0].size() - 8);
		if (origin_index != mask_index)
			throw exception("name don't match"); // 检查原图与皮肤图的编号是否匹配

		Mat origin = imread(origin_names[i]);
		allSetX.push_back(origin.reshape(origin.channels(), origin.total())); // 将图像尺寸转换成pixelNum*1再连成一长列
		Mat mask = imread(mask_names[i], 0);
		allSetY.push_back(mask.reshape(mask.channels(), mask.total()));
	}
	threshold(allSetY, allSetY, 1, 255, THRESH_BINARY);

	// 随机化像素顺序
	shufflePixel(allSetX, allSetY);

	return make_pair(allSetX, allSetY);
}

pair<Mat, Mat> readFullPicture() {
	// 读取前57个样本（尺寸相同）
	vector<string> origin_names, mask_names;
	readDirectory("original_images", origin_names, "jpg");
	readDirectory("skin_masks", mask_names, "jpg");
	printf("picNum: %d\n", origin_names.size());

	Mat allSetX, allSetY;
	for (unsigned i = 0; i < 57; i++) {
		auto origin_index = origin_names[i].substr(origin_names[0].size() - 8);
		auto mask_index = mask_names[i].substr(mask_names[0].size() - 8);
		if (origin_index != mask_index)
			throw exception("name don't match"); // 检查原图与皮肤图的编号是否匹配

		Mat origin = imread(origin_names[i]);
		allSetX.push_back(origin);
		Mat mask = imread(mask_names[i], 0);
		allSetY.push_back(mask);
	}
	threshold(allSetY, allSetY, 1, 255, THRESH_BINARY);

	return make_pair(allSetX, allSetY);
}

void extractForeground() {
	// Open the video file
	cv::VideoCapture capture(0);
	// check if video successfully opened
	if (!capture.isOpened())
		return;
	// current video frame
	cv::Mat frame;
	// foreground binary image
	cv::Mat foreground;
	cv::namedWindow("Extracted Foreground");
	// The Mixture of Gaussian object
	// used with all default parameters
	cv::BackgroundSubtractorMOG mog;
	bool stop(false);
	// for all frames in video
	while (!stop) {
		// read next frame if any
		if (!capture.read(frame))
			break;
		resize(frame, frame, Size(320, 240));
		// update the background
		// and return the foreground
		mog(frame, foreground, 0.005);
		// Complement the image
//		cv::threshold(foreground, foreground, 128, 255, cv::THRESH_BINARY);
		// show foreground
		cv::imshow("Extracted Foreground", foreground);
		// introduce a delay
		// or press key to stop
		if (cv::waitKey(10) >= 0)
			stop = true;
	}
}

/*
void filter(){
	Mat resultBin = imread("resultBin.jpg", 0);
	imshow("resultBin", resultBin);

	Mat blured;
	blur(resultBin, blured, Size(5, 5));
	threshold(blured, blured, 128, 255, THRESH_BINARY);
	imshow("Blur", blured);

	Mat gaussianBlur;
	GaussianBlur(resultBin, gaussianBlur, Size(5, 5), 10.0);
	threshold(gaussianBlur, gaussianBlur, 128, 255, THRESH_BINARY);
	imshow("GaussianBlur", gaussianBlur);

	Mat closeOpen;
	Mat element(3, 3, CV_8U, cv::Scalar(1));
	morphologyEx(resultBin, closeOpen, MORPH_CLOSE, element);
	morphologyEx(closeOpen, closeOpen, MORPH_OPEN, element);
	imshow("close-open", closeOpen);

	Mat median;
	medianBlur(resultBin, median, 5);
	imshow("medianBlur", median);
}
*/
