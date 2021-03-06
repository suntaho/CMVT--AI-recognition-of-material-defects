
//====================================================================
//
// 文件: CvxText.h
//
// 說明: OpenCV漢字輸出
//
// 時間: 
// 作者: chaishushan{AT}gmail.com
//
//====================================================================
//====================================================================

#ifndef OPENCV_CVX_TEXT_2007_08_31_H
#define OPENCV_CVX_TEXT_2007_08_31_H

//
// file CvxText.h
// brief OpenCV漢字輸出介面
// 實現了漢字輸出功能
//

#include <ft2build.h>
#include FT_FREETYPE_H
#include <opencv2/opencv.hpp>

//
// class CvxText
// brief OpenCV中輸出漢字
// OpenCV中輸出漢字.字形檔提取採用了開源的FreeFype庫.由於FreeFype是
// GPL版權發佈的庫,和OpenCV版權並不一致,因此目前還沒有合併到OpenCV
// 擴展庫中
// 顯示漢字的時候需要一個漢字字形檔檔,字形檔檔案系統一般都自帶了
// 這裡採用的是一個開源的字形檔:"文泉驛正黑體"
// 關於"OpenCV擴展庫"的細節請訪問
// http://code.google.com/p/opencv-extension-library/
//
// 關於FreeType的細節請訪問
// http://www.freetype.org/
//
// 例子:
//
/*
int main(int argc, char *argv[])
{
     // 定義CvxApplication物件
	 CvxApplication app(argc, argv);
     // 打開一個影像
	 IplImage *img = cvLoadImage("test.jpg", 1);
     // 輸出漢字
	 {
         // "wqy-zenhei.ttf"為文泉驛正黑體
		 CvText text("wqy-zenhei.ttf");
		 const char *msg = "在OpenCV中輸出漢字!";
		 float p = 0.5;
		 text.setFont(NULL, NULL, NULL, &p);        // 透明處理
		 text.putText(img, msg, cvPoint(100, 150), CV_RGB(255, 0, 0));
	 }
     // 定義視窗,並顯示影像
	 CvxWindow myWin("myWin");
	 myWin.showImage(img);
     // 進入消息迴圈
	 return app.exec();
 }
*/
class CvxText
{
    // 禁止copy
	CvxText& operator=(const CvxText&);
    //================================================================
public:
    // 裝載字形檔檔
	CvxText(const char *freeType);
	virtual ~CvxText();
	//================================================================
	//獲取字體。目前有些參數尚不支持
	//
	// param font        字體類型, 目前不支援
	// param size        字體大小 / 空白比例 / 間隔比例 / 旋轉角度
	// param underline   下畫線
	// param diaphaneity 透明度
	// sa setFont, restoreFont
	// 
	void getFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	// 設置字體.目前有些參數尚不支持
	//
	// param font        字體類型,目前不支援
	// param size        字體大小/空白比例/間隔比例/旋轉角度
	// param underline   下畫線
	// param diaphaneity 透明度
	// sa getFont, restoreFont
	void setFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	//
	// 恢復原始的字體設置
	//
	// getFont, setFont
	//
	void restoreFont();
	//================================================================
	//================================================================
	// 輸出漢字(顏色默認為黑色).遇到不能輸出的字元將停止.
	//
	// param img輸出的影像
	// param text 文本內容
	// param pos文本位置
	//
	// return 返回成功輸出的字元長度,失敗返回-1
	int putText(IplImage *img, const char    *text, CvPoint pos);
	// 輸出漢字(顏色默認為黑色).遇到不能輸出的字元將停止.
	//
	// param img輸出的影像
	// param text 文本內容
	// param pos文本位置
	//
	// return 返回成功輸出的字元長度,失敗返回-1
	int putText(IplImage *img, const wchar_t *text, CvPoint pos);


	// 輸出漢字.遇到不能輸出的字元將停止
	//
	// param img   輸出的影像
	// param text  文本內容
	// param pos   文本位置
	// param color 文本顏色
	//
	// return 返回成功輸出的字元長度,失敗返回-1
	int putText(IplImage *img, const char    *text, CvPoint pos, CvScalar color);
	// 輸出漢字.遇到不能輸出的字元將停止.
	//
	// param img   輸出的影像
	// param text  文本內容
	// param pos   文本位置
	// param color 文本顏色
	//
	// return 返回成功輸出的字元長度,失敗返回-1.
	int putText(IplImage *img, const wchar_t *text, CvPoint pos, CvScalar color);
	//================================================================
	//================================================================
private:
	// 輸出當前字元, 更新m_pos位置
	void putWChar(IplImage *img, wchar_t wc, CvPoint &pos, CvScalar color);
	//================================================================
	//================================================================
private:
	FT_Library        m_library;        // 字形檔
	FT_Face           m_face;           // 字體
	//================================================================
	//================================================================
	// 預設的字體輸出參數
	int               m_fontType;
	CvScalar          m_fontSize;
	bool              m_fontUnderline;
	float             m_fontDiaphaneity;
	//================================================================
	//================================================================
};
#endif // OPENCV_CVX_TEXT_2007_08_31_H
