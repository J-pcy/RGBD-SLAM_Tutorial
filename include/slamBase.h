# pragma once

#include <fstream>
#include <vector>
#include <map>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/core/eigen.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参结构
struct CAMERA_INTRINSIC_PARAMETERS
{
	double cx, cy, fx, fy, scale;
};

// 帧结构
struct FRAME
{
	cv::Mat rgb, depth;
	cv::Mat desp;
	vector<cv::KeyPoint> kp;
};

// PnP 结果
struct RESULT_OF_PNP
{
	cv::Mat rvec, tvec;
	int inliers;
};

// image2PonitCloud 将rgb图转换为点云
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

// point2dTo3d 将单个点从图像坐标转换为空间坐标
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camers);

// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp(FRAME& frame, string detector, string descriptor);

// estimateMotion 计算两个帧之间的运动
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

// cvMat2Eigen 将cv的旋转矢量与位移矢量转换为变换矩阵
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);

// joinPointCloud 将新的帧合并到旧的点云里
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera);

// 参数读取类
class ParameterReader
{
public:
	ParameterReader(string filename= "./parameters.txt")
	{
		ifstream fin(filename.c_str());
		if(!fin)
		{
			cerr <<"parameter file does not exist." <<endl;
			return;
		}
		while(!fin.eof())
		{
			string str;
			getline(fin, str);
			
			if(str[0]=='#')
				continue;

			int pos = str.find("=");
			if(pos==-1)
				continue;

			string key = str.substr(0, pos);
			string value = str.substr(pos+1, str.length());
			data[key] = value;

			if(!fin.good())
				break;
		}
	}

	string getData(string key)
	{
		map<string, string>::iterator iter = data.find(key);
		if(iter== data.end())
		{
			cerr << "Parameter name " <<key <<" not found!" <<endl;
			return string("NOT FOUND");
		}
		return iter -> second;
	}

public:
	map<string, string> data;
};

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
	ParameterReader pd;
	CAMERA_INTRINSIC_PARAMETERS camera;
	camera.fx = atof(pd.getData("camera.fx").c_str());
	camera.fy = atof(pd.getData("camera.fy").c_str());
	camera.cx = atof(pd.getData("camera.cx").c_str());
	camera.cy = atof(pd.getData("camera.cy").c_str());
	camera.scale = atof(pd.getData("camera.scale").c_str());
	return camera;
}
