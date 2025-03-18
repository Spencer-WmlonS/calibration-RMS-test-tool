#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main() {
    // 定义棋盘格尺寸
    Size boardSize(9, 6); // 棋盘格内角点数
    float squareSize = 0.008f; // 每个方格的实际尺寸

    // 生成棋盘格的 3D 坐标
    vector<Point3f> objectCorners;
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            objectCorners.emplace_back(j * squareSize, i * squareSize, 0);

    // 读取棋盘格图像
    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;

    vector<String> fn;
    glob(".\\chess1\\*.png", fn, false);

    for (int i = 0; i < fn.size(); ++i) { 
        printf("processing %d / %d\n",i+1,fn.size());
        Mat image = imread(fn[i]);
        if (image.empty()) {
            cerr << "无法读取图像文件: " << fn[i] << endl;
            continue;
        }
        Mat gray;
        //cvtColor(image, gray, COLOR_BGR2GRAY);
        gray=image;

        // 检测棋盘格角点
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            // 亚像素精确化
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));//其实俺也不知道这是什么

            // 保存角点数据
            imagePoints.push_back(corners);
            objectPoints.push_back(objectCorners);

            // 可视化角点
            drawChessboardCorners(image, boardSize, corners, found);
            imshow("Chessboard Corners", image);
            waitKey(20); 
        }
    }
    destroyAllWindows();

    // 读取标定文件
    cv::FileStorage fs("ost6.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "can not open file" << endl;
        return -1;
    }

    // 解析标定参数
    cv::Mat cameraMatrix, distCoeffs,camera_matrix,dist_coeffs;
    vector<Mat> rvecs, tvecs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    cout << "calculating...\n" << endl;
    //这句只是用来求rvecs和tvecs的(有什么更好的办法吗呜呜呜)
    calibrateCamera(objectPoints, imagePoints, boardSize, camera_matrix, dist_coeffs, rvecs, tvecs);
    cout << "camera matrix:\n" << camera_matrix << endl;
    cout << "distortion coefficients:\n" << dist_coeffs << endl;

    // 计算投影误差
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    vector<float> perViewErrors;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    double rms=sqrt(totalErr/totalPoints);

    cout << "重投影误差(RMS): " << rms << " 像素\n";
    fs.release();
    waitKey(0);
    return 0;
}