#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

FRAME readFrame(int index, ParameterReader& pd);

double normofTransform(cv::Mat rvec, cv::Mat tvec);

int main(int argc, char** argv)
{
	ParameterReader pd;
	int startIndex = atoi(pd.getData("start_index").c_str());
	int endIndex = atoi(pd.getData("end_index").c_str());

	cout <<"Initializing ..." <<endl;
	int currIndex = startIndex;
	FRAME lastFrame = readFrame(currIndex, pd);

	string detector = pd.getData("detector");
	string descriptor = pd.getData("descriptor");
	CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
	computeKeyPointsAndDesp(lastFrame, detector, descriptor);
	PointCloud::Ptr cloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);

	pcl::visualization::CloudViewer viewer("viewer");

	bool visualize = pd.getData("visualize_pointcloud") == string("yes");

	int min_inliers = atoi(pd.getData("min_inliers").c_str());
	double max_norm = atof(pd.getData("max_norm").c_str());

	/************g2o初始化**********************/
	typedef g2o::BlockSolver_6_3 SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

	SlamLinearSolver* linearSolver = new SlamLinearSolver();
	linearSolver -> setBlockOrdering(false);
	SlamBlockSolver* blockSolver = new SlamBlockSolver(std::unique_ptr<SlamLinearSolver>(linearSolver));
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<SlamBlockSolver>(blockSolver));

	g2o::SparseOptimizer globalOptimizer;
	globalOptimizer.setAlgorithm(solver);
	globalOptimizer.setVerbose(false);

	g2o::VertexSE3* v = new g2o::VertexSE3();
	v -> setId(currIndex);
	v -> setEstimate(Eigen::Isometry3d::Identity());
	v -> setFixed(true);
	globalOptimizer.addVertex(v);

	int lastIndex = currIndex;

	for(currIndex=startIndex+1; currIndex<endIndex; currIndex++)
	{
		cout <<"Reading files " << currIndex <<endl;
		FRAME currFrame = readFrame(currIndex, pd);
		computeKeyPointsAndDesp(currFrame, detector, descriptor);

		RESULT_OF_PNP result = estimateMotion(lastFrame, currFrame, camera);
		if(result.inliers < min_inliers)
		continue;

		double norm = normofTransform(result.rvec, result.tvec);
		cout <<"norm = " << norm <<endl;
		if(norm >= max_norm)
			continue;
		Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
		cout << "T = " << T.matrix() <<endl;

//		cloud = joinPointCloud(cloud, currFrame, T, camera);
		if(visualize == true)
		{
			cloud = joinPointCloud(cloud, currFrame, T, camera);
			viewer.showCloud(cloud);
		}

		g2o::VertexSE3* v = new g2o::VertexSE3();
		v -> setId(currIndex);
		v -> setEstimate(Eigen::Isometry3d::Identity());
		globalOptimizer.addVertex(v);

		g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		edge -> vertices()[0] = globalOptimizer.vertex(lastIndex);
		edge -> vertices()[1] = globalOptimizer.vertex(currIndex);

		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix <double, 6, 6>::Identity();
		information(0, 0) = information(1, 1) = information(2, 2) = 100;
		information(3, 3) = information(4, 4) = information(5, 5) = 100;

		edge -> setInformation(information);
		edge -> setMeasurement(T);
		globalOptimizer.addEdge(edge);

		lastFrame = currFrame;
		lastIndex = currIndex;
	}

//	pcl::io::savePCDFile("data/result.pcd", *cloud);
	cout << "optimizing pose graph, vertices: " << globalOptimizer.vertices().size() <<endl;
	globalOptimizer.save("./data/result_before.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(100);
	globalOptimizer.save("./data/result_after.g2o");
	cout <<"Optimization done." <<endl;

	globalOptimizer.clear();
	return 0;
}

FRAME readFrame(int index, ParameterReader& pd)
{
	FRAME f;
	string rgbDir = pd.getData("rgb_dir");
	string depthDir = pd.getData("depth_dir");
	string rgbExt = pd.getData("rgb_extension");
	string depthExt = pd.getData("depth_extension");

	stringstream ss;
	ss << rgbDir << index << rgbExt;
	string filename;
	ss >> filename;
	f.rgb = cv::imread(filename);

	ss.clear();
	filename.clear();
	ss << depthDir << index << depthExt;
	ss >> filename;
	f.depth = cv::imread(filename, -1);
	return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
	return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec))) + fabs(cv::norm(tvec));
}

