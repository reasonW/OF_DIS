#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>
    
#include "oflow.h"

using namespace cv;
using namespace std;
static vector<Scalar> colorwheel; //Scalar r,g,b
#define UNKNOWN_FLOW_THRESH 1e9

// Save a Depth/OF/SF as .flo file
void SaveFlowFile(cv::Mat& img, const char* filename);

// Save a depth as .pfm file
void SavePFMFile(cv::Mat& img, const char* filename);

// Read a depth/OF/SF as file
void ReadFlowFile(cv::Mat& img, const char* filename)
{
  FILE *stream = fopen(filename, "rb");
  if (stream == 0)
    cout << "ReadFile: could not open %s" << endl;
  
  int width, height;
  float tag;
  int nc = img.channels();
  float tmp[nc];  

  if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
      (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
      (int)fread(&height, sizeof(int),   1, stream) != 1)
        cout << "ReadFile: problem reading file %s" << endl;

  for (int y = 0; y < height; y++) 
  {
    for (int x = 0; x < width; x++) 
    {
      if ((int)fread(tmp, sizeof(float), nc, stream) != nc)
        cout << "ReadFile(%s): file is too short" << endl;

      if (nc==1) // depth
        img.at<float>(y,x) = tmp[0];
      else if (nc==2) // Optical Flow
      {
        img.at<cv::Vec2f>(y,x)[0] = tmp[0];
        img.at<cv::Vec2f>(y,x)[1] = tmp[1];
      }
      else if (nc==4) // Scene Flow
      {
        img.at<cv::Vec4f>(y,x)[0] = tmp[0];
        img.at<cv::Vec4f>(y,x)[1] = tmp[1];
        img.at<cv::Vec4f>(y,x)[2] = tmp[2];
        img.at<cv::Vec4f>(y,x)[3] = tmp[3];
      }
    }
  }

  if (fgetc(stream) != EOF)
    cout << "ReadFile(%s): file is too long" << endl;

  fclose(stream);
}

void ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype, const bool getgrad, const int imgpadding, const int padw, const int padh)
{
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      if (i==0) // At finest scale: copy directly, for all other: downscale previous scale by .5
      {
        img_ao_fmat_pyr[i] = img_ao_fmat.clone();
      }
      else
        cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
	      
      img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);
	
      if ( getgrad ) 
      {
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
        img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
        img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
      }
    }
    
    // pad images
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
      img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

      if ( getgrad ) 
      {
        copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0); // Zero padding for gradients
        copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0);

        img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
        img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;      
      }
    }
}

int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize)
{
  return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth) / ((float)fratio * (float)patchsize))));
}

//opencv coloe BGR system
void makecolorwheel(vector<Scalar> &colorwheel)  
{  //共55个方向
	int RY = 15;  
	int YG = 6;  
	int GC = 4;  
	int CB = 11;  
	int BM = 13;  
	int MR = 6;  
	
	int i;  
	
	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  

void motionToColor(const Mat flow,Mat &color)  
{ 
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  

    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
    // determine motion range:  
    float maxrad = -1;  
    // Mat rad;
    // rad.create(flowx.rows, flowx.cols,flowx.type());	    	
    //Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            int rad=sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  

	#pragma omp parallel for 
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;   
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
  
            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
             for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
}

int main( int argc, char** argv )
{
   	VideoCapture cap;
	cap.open(0);
    if( !cap.isOpened() )
        		return -1;
	Mat pre_img, post_img, frame;
    while(1)
    {
    	cap>>pre_img;
    	if( post_img.data )
    	{
			// *** Parse and load input images
			cv::Mat img_ao_mat, img_bo_mat, img_tmp;
			int rpyrtype, nochannels, incoltype;
			incoltype = CV_LOAD_IMAGE_COLOR;
			rpyrtype = CV_32FC3;
			nochannels = 3;      

			img_ao_mat = pre_img;   // Read the file
			img_bo_mat = post_img;   // Read the file    
			cv::Mat img_ao_fmat, img_bo_fmat;
			cv::Size sz = img_ao_mat.size();
			int width_org = sz.width;   // unpadded original image size
			int height_org = sz.height;  // unpadded original image size 

			// *** Parse rest of parameters, See oflow.h for definitions.
			int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
			float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
			bool usefbcon, usetvref;
			//bool hasinfile; // initialization flow file
			//char *infile = nullptr;

			mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;    
			usefbcon = 0; patnorm = 1; costfct = 0; 
			tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
			tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
			verbosity = 2; // Default: Plot detailed timings
			    
			int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.
			int sel_oppoint = 2; // Default operating point

			switch (sel_oppoint)
			{
			  case 1:
			    patchsz = 8; poverl = 0.3; 
			    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
			    lv_l = std::max(lv_f-2,0); maxiter = 16; miniter = 16; 
			    usetvref = 0; 
			    break;
			  case 3:
			    patchsz = 12; poverl = 0.75; 
			    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
			    lv_l = std::max(lv_f-4,0); maxiter = 16; miniter = 16; 
			    usetvref = 1; 
			    break;
			  case 4:
			    patchsz = 12; poverl = 0.75; 
			    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
			    lv_l = std::max(lv_f-5,0); maxiter = 128; miniter = 128; 
			    usetvref = 1; 
			    break;        
			  case 2:
			  default:
			    patchsz = 8; poverl = 0.4; 
			    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
			    lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12; 
			    usetvref = 1; 
			    break;

			}
			// *** expand(pad) image such that width and height are restless divisible on all scales (except last)
			int padw=0, padh=0;
			int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
			//if (hasinfile) scfct = pow(2,lv_f+1); // if initialization file is given, make sure that size is restless divisible by 2^(lv_f+1) !
			int div = sz.width % scfct;
			if (div>0) padw = scfct - div;
			div = sz.height % scfct;
			if (div>0) padh = scfct - div;          
			if (padh>0 || padw>0)
			{
			copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
			copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
			}
			sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
			//  *** Generate scale pyramides (金字塔)
			img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
			img_bo_mat.convertTo(img_bo_fmat, CV_32F);
			const float* img_ao_pyr[lv_f+1];
			const float* img_bo_pyr[lv_f+1];
			const float* img_ao_dx_pyr[lv_f+1];
			const float* img_ao_dy_pyr[lv_f+1];
			const float* img_bo_dx_pyr[lv_f+1];
			const float* img_bo_dy_pyr[lv_f+1];

			cv::Mat img_ao_fmat_pyr[lv_f+1];
			cv::Mat img_bo_fmat_pyr[lv_f+1];
			cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
			cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
			cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
			cv::Mat img_bo_dy_fmat_pyr[lv_f+1];

			ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr,\
			lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
			ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, \
			lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);

			//  *** Run main optical flow / depth algorithm
			float sc_fct = pow(2,lv_l);
			cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC2); // Optical Flow
 
			OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, 
			                img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, 
			                patchsz,  // extra image padding to avoid border violation check
			                (float*)flowout.data,   // pointer to n-band output float array
			                nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
			                sz.width, sz.height, 
			                lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, 
			                usefbcon, costfct, nochannels, patnorm, 
			                usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
			                verbosity);    

			// *** Resize to original scale, if not run to finest level
			if (lv_l != 0)
			{
			flowout *= sc_fct;
			cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
			}

			// If image was padded, remove padding before saving to file
			cv::Rect ipad=cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org);
			flowout = flowout(ipad);
			Mat flow_show;
			motionToColor(flowout,flow_show);
            imshow("flow",flow_show);
            char key=waitKey(1);
            if (key=='q')
            	return 0;
	    }
      	cv::swap(post_img, pre_img);	
	}


}

void SaveFlowFile(cv::Mat& img, const char* filename)
{
  cv::Size szt = img.size();
  int width = szt.width, height = szt.height;
  int nc = img.channels();
  float tmp[nc];

  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
    cout << "WriteFile: could not open file" << endl;

  // write the header
  fprintf(stream, "PIEH");
  if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
      (int)fwrite(&height, sizeof(int),   1, stream) != 1)
    cout << "WriteFile: problem writing header" << endl;

  for (int y = 0; y < height; y++) 
  {
    for (int x = 0; x < width; x++) 
    {
      if (nc==1) // depth
        tmp[0] = img.at<float>(y,x);
      else if (nc==2) // Optical Flow
      {nvi
        tmp[0] = img.at<cv::Vec2f>(y,x)[0];
        tmp[1] = img.at<cv::Vec2f>(y,x)[1];
      }
      else if (nc==4) // Scene Flow
      {
        tmp[0] = img.at<cv::Vec4f>(y,x)[0];
        tmp[1] = img.at<cv::Vec4f>(y,x)[1];
        tmp[2] = img.at<cv::Vec4f>(y,x)[2];
        tmp[3] = img.at<cv::Vec4f>(y,x)[3];
      }	  

      if ((int)fwrite(tmp, sizeof(float), nc, stream) != nc)
        cout << "WriteFile: problem writing data" << endl;         
    }
  }
  fclose(stream);
}
    
// Save a depth as .pfm file
void SavePFMFile(cv::Mat& img, const char* filename)
{
  cv::Size szt = img.size();
  
  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
    cout << "WriteFile: could not open file" << endl;

  // write the header
  fprintf(stream, "Pf\n%d %d\n%f\n", szt.width, szt.height, (float)-1.0f);    
  
  for (int y = szt.height-1; y >= 0 ; --y) 
  {
    for (int x = 0; x < szt.width; ++x) 
    {
      float tmp = -img.at<float>(y,x);
      if ((int)fwrite(&tmp, sizeof(float), 1, stream) != 1)
        cout << "WriteFile: problem writing data" << endl;         
    }
  }  
  fclose(stream);
}

