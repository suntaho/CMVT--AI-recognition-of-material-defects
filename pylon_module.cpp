#include "cmvt_ai.h"


// relevant variables
Pylon::PylonAutoInitTerm autoInitTerm;                                 // PylonInitialize and PylonTerminate
Camera_t camera;                                                 // Create an object with the camera device found first
CImageFormatConverter formatConverter;                                 // Create a Pylon ImageFormatConverter object
CPylonImage PylonImage;                                                // PylonImage
CGrabResultPtr ptrGrabResult;                                          // pointer to receive the grab result
cv::Size PylonImageSize;                                               // size of Pylon image
static const uint32_t c_countOfImagesToGrab = -1;                      // free-running continuous acquisition


// subroutines
// initialize Pylon camera	
void Pylon_ini() {
	camera.Attach(CTlFactory::GetInstance().CreateFirstDevice());
	cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;                      // show message of device
	GenApi::INodeMap& nodemap = camera.GetNodeMap();                                               // Get a camera nodemap in order to access camera parameters
	camera.Open();
	GenApi::CIntegerPtr pwidth = nodemap.GetNode("Width");
	GenApi::CIntegerPtr pheight = nodemap.GetNode("Height");
	formatConverter.OutputPixelFormat = PixelType_Mono8;
	PylonImageSize = Size((int)pwidth->GetValue(), (int)pheight->GetValue());
}

// adjust camera's widht/hgieht
void Pylon_setWH(int64_t newWidth, int64_t newHeight) {
	if (newWidth < camera.Width.GetMin()) newWidth = camera.Width.GetMin();
	if (newWidth > camera.Width.GetMax()) newWidth = camera.Width.GetMax();
	if (newHeight < camera.Height.GetMin()) newHeight = camera.Height.GetMin();
	if (newHeight > camera.Height.GetMax()) newHeight = camera.Height.GetMax();
	// images size
	camera.Width.SetValue(newWidth);
	camera.Height.SetValue(newHeight);
	PylonImageSize = Size((int)newWidth, (int)newHeight);
	cout << "adjust image Width/Height: " << newWidth << " / " << newHeight << endl;
}

// adjust camera's Exposure-Time (us)
void Pylon_setET(double exposuretime) {
	if (exposuretime < 0.0) exposuretime = 1.0;
	if (exposuretime > 250000.0) exposuretime = 250000.0;
	// exposure time
	camera.ExposureTimeAbs.SetValue(exposuretime);
}

// start camera
void Pylon_begin() { 
	camera.StartGrabbing(c_countOfImagesToGrab, GrabStrategy_LatestImageOnly);                     // start the grabbing of Pylon images
}  

// request the grabbing-image status 
bool Pylon_IsGrabSuceed() {
	camera.RetrieveResult(3000, ptrGrabResult, TimeoutHandling_ThrowException);                    // wait for an image and then retrieve it. A timeout of 5000 ms is used.
	if (camera.IsGrabbing() && ptrGrabResult->GrabSucceeded()) return true;
	else return false;
}

// convert Pylon image to OpenCV image
Mat Pylon2Mat() {
	Mat PylonImage_buf;
	formatConverter.Convert(PylonImage, ptrGrabResult);                                            // convert the grabbed buffer to a pylon image
	PylonImage_buf = cv::Mat(PylonImageSize.height, PylonImageSize.width, CV_8UC1,                 // create an opencv image from a pylon image
		(uint8_t *)PylonImage.GetBuffer());
	return PylonImage_buf;
}

// close device
void Pylon_close() {
	camera.StopGrabbing();
	camera.Close();
}