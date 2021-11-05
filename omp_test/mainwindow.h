#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <iostream>
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QThread>
#include <QGridLayout>
#include <QDir>
#include <opencv2/opencv.hpp>
#include "yolo_v2_class.hpp"
#include "featuremodel.h"
#include "DeepSort/matching/tracker.h"

#define args_nn_budget 100
#define args_max_cosine_distance 0.2
#define args_min_confidence 0.3
#define args_nms_max_overlap 1.0


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void section1();
    void section2();
    void section3();

    QPushButton *btnTest;
    QLabel *lblImage;

    Detector *detector;
    FeatureModel *featureModel;
    tracker *mytracker;

    ImageFrame frames[3];
    cv::Mat images[3];
    int counter = 0;

    std::string cfg_filename = "C:\\Users\\sr996\\models\\reduced\\ami1\\yolov4.cfg";
    std::string weight_filename = "C:\\Users\\sr996\\models\\reduced\\ami1\\yolov4.weights";
    QString saved_model_dir = "C:\\Users\\sr996\\source\\repos\\deep_sort_v2\\saved_model";
    QString image_dir = "C:\\Users\\sr996\\source\\repos\\deep_sort_v2\\MOT16\\test\\MOT16-06\\img1\\";
    QStringList image_filenames;


public slots:
    void test();

};
#endif // MAINWINDOW_H
