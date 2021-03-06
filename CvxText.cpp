
#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include <ctype.h>

#include "CvxText.h"

//====================================================================
//====================================================================

// 打開字形檔

CvxText::CvxText(const char *freeType)
{
	assert(freeType != NULL);
	// 打開字形檔檔, 創建一個字體
	if (FT_Init_FreeType(&m_library)) throw;
	if (FT_New_Face(m_library, freeType, 0, &m_face)) throw;
	// 設置字體輸出參數
	restoreFont();
	// 設置C語言的字元集環境
	setlocale(LC_ALL, "");
}
// 釋放FreeType資源
CvxText::~CvxText()
{
	FT_Done_Face(m_face);
	FT_Done_FreeType(m_library);
}

// 設置字體參數:
//
// font             - 字體類型, 目前不支援
// size             - 字體大小/空白比例/間隔比例/旋轉角度
// underline        - 下畫線
// diaphaneity      - 透明度
void CvxText::getFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
	if (type) *type = m_fontType;
	if (size) *size = m_fontSize;
	if (underline) *underline = m_fontUnderline;
	if (diaphaneity) *diaphaneity = m_fontDiaphaneity;
}

void CvxText::setFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
	// 參數合法性檢查
	if (type)
	{
		if (type >= 0) m_fontType = *type;
	}
	if (size)
	{
		m_fontSize.val[0] = fabs(size->val[0]);
		m_fontSize.val[1] = fabs(size->val[1]);
		m_fontSize.val[2] = fabs(size->val[2]);
		m_fontSize.val[3] = fabs(size->val[3]);
	}
	if (underline)
	{
		m_fontUnderline = *underline;
	}
	if (diaphaneity)
	{
		m_fontDiaphaneity = *diaphaneity;
	}
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}
// 恢復原始的字體設置
void CvxText::restoreFont()
{
	m_fontType = 0;                 // 字體類型(不支援)
	m_fontSize.val[0] = 20;         // 字體大小
	m_fontSize.val[1] = 0.5;        // 空白字元大小比例
	m_fontSize.val[2] = 0.1;        // 間隔大小比例
	m_fontSize.val[3] = 0;          // 旋轉角度(不支持)
	m_fontUnderline = false;        // 下畫線(不支持)
	m_fontDiaphaneity = 1.0;        // 色彩比例(可產生透明效果)
        // 設置字元大小
	FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}
// 輸出函數(顏色預設為黑色)
int CvxText::putText(IplImage *img, const char *text, CvPoint pos)
{
	return putText(img, text, pos, CV_RGB(255, 255, 255));
}
int CvxText::putText(IplImage *img, const wchar_t *text, CvPoint pos)
{
	return putText(img, text, pos, CV_RGB(255, 255, 255));
}
//
int CvxText::putText(IplImage *img, const char    *text, CvPoint pos, CvScalar color)
{
	if (img == NULL) return -1;
	if (text == NULL) return -1;
    //
	int i;
	for (i = 0; text[i] != '\0'; ++i)
	{
		wchar_t wc = text[i];
        // 解析雙位元組符號
		if (!isascii(wc)) mbtowc(&wc, &text[i++], 2);
        // 輸出當前的字元
		putWChar(img, wc, pos, color);
	}
	return i;
}
int CvxText::putText(IplImage *img, const wchar_t *text, CvPoint pos, CvScalar color)
{
	if (img == NULL) return -1;
	if (text == NULL) return -1;
    //
	int i;
	for (i = 0; text[i] != '\0'; ++i)
	{
		// 輸出當前的字元
		putWChar(img, text[i], pos, color);
	}
	return i;
}
// 輸出當前字元, 更新m_pos位置
void CvxText::putWChar(IplImage *img, wchar_t wc, CvPoint &pos, CvScalar color)
{
	// 根據unicode生成字體的二值點陣圖
	FT_UInt glyph_index = FT_Get_Char_Index(m_face, wc);
	FT_Load_Glyph(m_face, glyph_index, FT_LOAD_DEFAULT);
	FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_MONO);
    //
	FT_GlyphSlot slot = m_face->glyph;
    // 行列數
	int rows = slot->bitmap.rows;
	int cols = slot->bitmap.width;
    //
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			int off = ((img->origin == 0) ? i : (rows - 1 - i)) *slot->bitmap.pitch + j / 8;
			if (slot->bitmap.buffer[off] & (0xC0 >> (j % 8)))
			{
				int r = (img->origin == 0) ? pos.y - (rows - 1 - i) : pos.y + i;
				int c = pos.x + j;
				if (r >= 0 && r < img->height && c >= 0 && c < img->width)
				{
					CvScalar scalar = cvGet2D(img, r, c);
					// 進行色彩融合
					float p = m_fontDiaphaneity;
					for (int k = 0; k < 4; ++k)
					{
						scalar.val[k] = scalar.val[k] * (1 - p) + color.val[k] * p;
					}
					cvSet2D(img, r, c, scalar);
				}
			}
		} // end for
	} // end for
        // 修改下一個字的輸出位置
	double space = m_fontSize.val[0] * m_fontSize.val[1];
	double sep = m_fontSize.val[0] * m_fontSize.val[2];
	pos.x += (int)((cols ? cols : space) + sep);
}
