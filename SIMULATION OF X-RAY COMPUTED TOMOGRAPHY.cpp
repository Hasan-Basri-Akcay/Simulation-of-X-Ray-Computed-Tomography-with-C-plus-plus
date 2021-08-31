// SIMULATION OF X-RAY COMPUTED TOMOGRAPHY.cpp : Bu dosya 'main' işlevi içeriyor. Program yürütme orada başlayıp biter.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>


using namespace std;
using namespace cv;


void fftshift(const Mat& input_img, Mat& output_img)
{
    output_img = input_img.clone();
    int cx = output_img.cols / 2;
    int cy = output_img.rows / 2;
    Mat q1(output_img, Rect(0, 0, cx, cy));
    Mat q2(output_img, Rect(cx, 0, cx, cy));
    Mat q3(output_img, Rect(0, cy, cx, cy));
    Mat q4(output_img, Rect(cx, cy, cx, cy));

    Mat temp;
    q1.copyTo(temp);
    q4.copyTo(q1);
    temp.copyTo(q4);
    q2.copyTo(temp);
    q3.copyTo(q2);
    temp.copyTo(q3);
}


void calculateDFT(Mat& scr, Mat& dst)
{
    // define mat consists of two mat, one for real values and the other for complex values
    scr.convertTo(scr, CV_32F);
    Mat planes[] = { scr, Mat::zeros(scr.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);

    dft(complexImg, complexImg);
    dst = complexImg;
}


Mat filter(Mat& scr, String type) {
    Mat H(scr.size(), CV_32F, Scalar(1));
    float D = 0;
    if (type == "ramp") {
        for (int u = 0; u < H.rows; u++) {
            for (int v = 0; v < H.cols; v++) {
                D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));

                H.at<float>(u, v) = D / (scr.rows / 2);
            }
        }
        return H;
    }
    else {
        return H;
    }
    
}


void filtering(Mat& scr, Mat& dst, Mat& H) {
    fftshift(H, H);
    Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

    Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
    split(scr, planes_dft);

    Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
    planes_out[0] = planesH[0].mul(planes_dft[0]);
    planes_out[1] = planesH[1].mul(planes_dft[1]);

    merge(planes_out, 2, dst);
}


void img_projection(string img_path, int number_projections, int number_sampling, int which_projection_show) {
    // Read the img according to img_path
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    Mat dst_sized_main;

    // Check the img. If it is empty, return null. Else continue.
    if (img.empty())
    {
        cout << "Could not read the image: " << img_path << endl;
    }
    else {
        // Calculate the rotation angle according to number_projections and push_back the rotation_angle_vec vector.
        vector<int>  rotation_angle_vec;
        for (int i = 0; i < number_projections; i++) {
            rotation_angle_vec.push_back(int((double(180) / number_projections) * i));
        }

        // Continue to prejections calculation
        vector<vector<int>>  projections;
        vector<vector<int>>  projections_norm;
        for (int i = 0; i < number_projections; i++) {
            // Rotate the img according to rotation_angle_vec parameters.
            Mat dst;
            Point2f pc(img.cols / 2., img.rows / 2.);
            Mat r = getRotationMatrix2D(pc, rotation_angle_vec[i], 1.0);
            warpAffine(img, dst, r, img.size());

            // Resize the img according to number_sampling. If img will grown, use INTER_CUBIC. Else use INTER_AREA.
            // INTER_CUBIC is good for resize img when img will grown. INTER_AREA is good for resize img when img will reduce.
            Mat dst_sized;
            Size size(number_sampling, number_sampling);
            if (img.cols <= number_sampling) {
                resize(dst, dst_sized, size, INTER_CUBIC);
                if (i == 0) {
                    dst_sized_main = dst_sized;
                }
            }
            else {
                resize(dst, dst_sized, size, INTER_AREA);
                if (i == 0) {
                    dst_sized_main = dst_sized;
                }
            }

            // In this loop, img rows push_back the row_vec.
            vector<vector<int>>  row_vec;
            for (int j = 0; j < dst_sized.rows; j++) {
                row_vec.push_back(dst_sized.row(j));
            }

            // Calculate the pixels value sum of the rows and push_back the projection vector. Then calculate projection_min and projection_max for plot the graph.
            vector<int> projection;
            for (int j = 0; j < dst_sized.rows; j++) {
                int sum = 0;
                for (int k = 0; k < dst_sized.cols; k++) {
                    sum += row_vec[j][k];
                }
                projection.push_back(sum);
            }

            // Push_back the projection to projections vector.
            projections.push_back(projection);

            // Show dst
            /*if (i == which_projection_show) {
                imshow("dst", dst);
            }*/
        }

        // Min max calculation for normalization
        int projection_min = 999999;
        int projection_max = -1;
        for (int i = 0; i < number_projections; i++) {
            for (int j = 0; j < number_sampling; j++) {
                if (projections[i][j] < projection_min) {
                    projection_min = projections[i][j];
                }
                if (projections[i][j] > projection_max) {
                    projection_max = projections[i][j];
                }
            }
        }

        // Projection normalization and plot graph.
        Mat plot_projection;
        int plot_size = 700;
        if (plot_size >= number_sampling) {
            Mat plot(plot_size, plot_size, CV_8UC1, Scalar(0));
            for (int i = 0; i < number_projections; i++) {
                vector<int> projection_norm;
                for (int j = 0; j < number_sampling; j++) {
                    double sum_norm = double(double(double(projections[i][j] - projection_min) / double(projection_max - projection_min)) - 0.5);
                    sum_norm = sum_norm * 600;
                    sum_norm = int(sum_norm + 350);
                    projection_norm.push_back(int(sum_norm));
                }
                double plot_step = double(double(plot_size) / number_sampling);
                for (int j = 0; j < number_sampling - 1; j++) {
                    line(plot, Point(int(double(j) * plot_step), plot_size - projection_norm[j]), Point(int((double(j) + 1) * plot_step), plot_size - projection_norm[j + 1]), Scalar(255), 1);
                }
                if (i == which_projection_show) {
                    Mat temp_plot(plot_size, plot_size, CV_8UC1, Scalar(0));

                    for (int j = 0; j < number_sampling - 1; j++) {
                        line(temp_plot, Point(int(double(j) * plot_step), plot_size - projection_norm[j]), Point(int((double(j) + 1) * plot_step), plot_size - projection_norm[j + 1]), Scalar(255), 1);
                    }

                    //imshow("projection", temp_plot);
                    //waitKey(0);
                }
            }
            plot_projection = plot;
        }
        else {
            int plot_size = number_sampling;
            Mat plot(plot_size, plot_size, CV_8UC1, Scalar(0));
            for (int i = 0; i < number_projections; i++) {
                vector<int> projection_norm;
                for (int j = 0; j < number_sampling; j++) {
                    double sum_norm = double(double(double(projections[i][j] - projection_min) / double(projection_max - projection_min)) - 0.5);
                    sum_norm = sum_norm * (plot_size - 200);
                    sum_norm = int(sum_norm + double(plot_size) / 2);
                    projection_norm.push_back(int(sum_norm));
                }
                double plot_step = double(double(plot_size) / number_sampling);
                for (int j = 0; j < number_sampling - 1; j++) {
                    line(plot, Point(int(double(j) * plot_step), plot_size - projection_norm[j]), Point(int((double(j) + 1) * plot_step), plot_size - projection_norm[j + 1]), Scalar(255), 1);
                }
                if (i == which_projection_show) {
                    Mat temp_plot(plot_size, plot_size, CV_8UC1, Scalar(0));

                    for (int j = 0; j < number_sampling - 1; j++) {
                        line(temp_plot, Point(int(double(j) * plot_step), plot_size - projection_norm[j]), Point(int((double(j) + 1) * plot_step), plot_size - projection_norm[j + 1]), Scalar(255), 1);
                    }

                    //imshow("projection", temp_plot);
                    //waitKey(0);
                }
            }
            plot_projection = plot;
        }

        // Save plot_projection
        string file_name = "Projection_data.png";
        imwrite(file_name, plot_projection);

        // Write the projections vector to projections.txt file.
        ofstream output_file("./projections.txt");
        output_file << number_projections << " " << number_sampling << endl;

        for (int i = 0; i < number_projections; i++) {
            for (int j = 0; j < number_sampling; j++) {
                output_file << projections[i][j] << " ";
            }
        }
        output_file.close();

        //plot sinogram
        Mat sinogram(number_sampling, number_projections, CV_8UC1, Scalar(0));
        for (int i = 0; i < number_projections; i++) {
            for (int j = 0; j < number_sampling; j++) {
                sinogram.at<char>(j, i) = int(double(projections[i][j] - projection_min) / double(projection_max - projection_min) * 255);
            }
        }
        imwrite("sinogram.png", sinogram);

        imshow("img", dst_sized_main);
        imshow("plot_projection", plot_projection);
        imshow("sinogram", sinogram);
        //waitKey(0);
    }
}


void img_back_projection(int which_projection_show) {
    // Read projection txt file
    ifstream infile("projections.txt");
    int number_projections, number_sampling;

    infile >> number_projections >> number_sampling;

    cout << "number_projections: " << number_projections << endl;
    cout << "number_sampling: " << number_sampling << endl;

    // Fill the projections vector with projections
    vector<vector<int>>  projections;
    for (int i = 0; i < number_projections; i++) {
        vector<int> projection;
        for (int j = 0; j < number_sampling; j++) {
            int a;
            infile >> a;
            projection.push_back(a);
        }
        projections.push_back(projection);
    }

    // Calculate the rotation angle according to number_projections and push_back the rotation_angle_vec vector.
    vector<int>  rotation_angle_vec;
    for (int i = 0; i < number_projections; i++) {
        rotation_angle_vec.push_back(int((double(180) / number_projections) * i));
    }

    // Rotate the projections
    vector<Mat>  img_vec;
    Mat which_projection_show_img;
    for (int i = 0; i < number_projections; i++) {
        Mat src(number_sampling, number_sampling, CV_32FC1, Scalar(0));
        int min_value = 99999;

        for (int j = 0; j < number_sampling; j++) {
            for (int k = 0; k < number_sampling; k++) {
                src.at<float>(k, j) = projections[i][k];
            }
        }

        if (src.at<float>(0, 0) <= src.at<float>(0, number_sampling - 1)) {
            min_value = src.at<float>(0, 0);
        }
        else {
            min_value = src.at<float>(0, number_sampling - 1);
        }

        // BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_TRANSPARENT
        Mat dst;
        Point2f pc(src.cols / 2., src.rows / 2.);
        Mat r = getRotationMatrix2D(pc, 90, 1);
        warpAffine(src, dst, r, src.size(), INTER_LINEAR, BORDER_REPLICATE, Scalar(min_value));

        // DFT
        dst.convertTo(dst, CV_32F);
        Mat DFT_image;
        calculateDFT(dst, DFT_image);
        // Calculate high pass filter
        Mat H;
        H = filter(DFT_image, "ramp");
        // Filtering
        Mat complexIH;
        filtering(DFT_image, complexIH, H);
        // IDFT
        Mat imgOut;
        dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);
        // Norm
        Mat imgOut_norm;
        normalize(imgOut, imgOut_norm, 0, 1, cv::NORM_MINMAX);
        Mat dst_norm;
        normalize(dst, dst_norm, 0, 1, cv::NORM_MINMAX);

        // BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_TRANSPARENT
        Mat dst_filtered_out;
        Point2f pc_2(imgOut.cols / 2., imgOut.rows / 2.);
        Mat r_2 = getRotationMatrix2D(pc_2, -rotation_angle_vec[i] - 90, 1);
        warpAffine(imgOut, dst_filtered_out, r_2, imgOut.size(), INTER_LINEAR, BORDER_REPLICATE, Scalar(min_value));
        Mat dst_filtered_out_norm;
        normalize(dst_filtered_out, dst_filtered_out_norm, 0, 1, cv::NORM_MINMAX);

        /*imshow("imgOut_norm", imgOut_norm);
        imshow("dst_norm", dst_norm);
        imshow("dst_filtered_out_norm", dst_filtered_out_norm);
        waitKey(0);*/

        if (i == which_projection_show) {
            which_projection_show_img = dst_filtered_out_norm;
        }

        img_vec.push_back(dst_filtered_out);
    }

    // Sum the projections and push back the sum_projections vector
    vector<vector<float>> sum_projections(number_sampling, vector<float>(number_sampling, 0));
    for (int i = 0; i < number_projections; i++) {
        for (int j = 0; j < number_sampling; j++) {
            for (int k = 0; k < number_sampling; k++) {
                sum_projections[k][j] = sum_projections[k][j] + img_vec[i].at<float>(k, j);
            }
        }
    }

    // Min max calculation for normalization
    float back_projection_min = 9999;
    float back_projection_max = -1;
    for (int i = 0; i < number_sampling; i++) {
        for (int j = 0; j < number_sampling; j++) {
            if (sum_projections[i][j] < back_projection_min) {
                back_projection_min = sum_projections[i][j];
            }
            if (sum_projections[i][j] > back_projection_max) {
                back_projection_max = sum_projections[i][j];
            }
        }
    }

    // back_projection_img normalization
    Mat back_projection_norm_img(number_sampling, number_sampling, CV_8UC1, Scalar(0));
    for (int j = 0; j < number_sampling; j++) {
        for (int k = 0; k < number_sampling; k++) {
            back_projection_norm_img.at<uchar>(k, j) = int(float(sum_projections[k][j] - back_projection_min) / (back_projection_max - back_projection_min) * 255);
        }
    }

    // Show and save back_projection_norm_img
    imshow("which_projection_show_img", which_projection_show_img);
    imwrite("back_projection_norm_img.png", back_projection_norm_img);
    imshow("back_projection_norm_img", back_projection_norm_img);
    waitKey(0);
}


int main()
{
    cout << "Please Enter the Function Parameters..." << endl;
    cout << "When asking img_path, press 'q' for exit." << endl;
    cout << "img_path(string)" << endl;
    cout << "number_projections(int)" << endl;
    cout << "number_sampling(int)" << endl;
    cout << "which_projection_show(int)" << endl << endl;
    while (true) {
        string img_path;
        cout << "img_path: ";
        cin >> img_path;

        if (img_path == "q") {
            break;
        }

        int number_projections;
        cout << "number_projections: ";
        cin >> number_projections;

        int number_sampling;
        cout << "number_sampling: ";
        cin >> number_sampling;

        int which_projection_show;
        cout << "which_projection_show: ";
        cin >> which_projection_show;

        // Call the img_projection function
        img_projection(img_path, number_projections, number_sampling, which_projection_show);

        // Call the img_back_projection function
        img_back_projection(which_projection_show);

        cout << endl;
    }

    return 0;
}
