#include "mainwindow.h"
#include "DeepSort/feature/model.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    btnTest = new QPushButton("test");
    connect(btnTest, SIGNAL(clicked()), this, SLOT(test()));
    lblImage = new QLabel();
    lblImage->setMinimumWidth(640);
    lblImage->setMinimumHeight(480);
    QWidget *panel = new QWidget();
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(lblImage, 0, 0, 1, 1);
    layout->addWidget(btnTest,  1, 0, 1, 1);
    panel->setLayout(layout);
    setCentralWidget(panel);

    image_filenames = QDir(image_dir).entryList(QDir::Files, QDir::Name);

    detector = new Detector(cfg_filename, weight_filename);
    featureModel = new FeatureModel();
    featureModel->load(saved_model_dir);
    mytracker = new tracker((float)args_max_cosine_distance, args_nn_budget);
}

void MainWindow::test()
{
    std::cout << "test" << std::endl;

    if (counter >= image_filenames.size()) {
        std::cout << "end of images" << std::endl;
        return;
    }

    images[2] = images[1];
    images[1] = images[0];

    frames[2] = frames[1];
    frames[1] = frames[0];


#pragma omp parallel sections
    {
#pragma omp section
        {
            section1();
        }
#pragma omp section
        {
            section2();
        }
#pragma omp section
        {
            section3();
        }
    }

    std::cout << "parallel end" << std::endl;
    counter++;
}

void MainWindow::section1()
{
    auto start = std::chrono::high_resolution_clock::now();
    //image = cv::imread(image_filename.toStdString());
    QString image_filename = image_dir + image_filenames[counter];
    images[0] = cv::imread(image_filename.toStdString());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "section 1 duration: " << QString::number(duration.count()).toStdString() << std::endl;
}

void MainWindow::section2()
{
    auto start = std::chrono::high_resolution_clock::now();
    if (images[1].rows > 0) {
        //std::cout << "images[1].rows: " << images[1].rows << " cols: " << images[1].cols << std::endl;
        frames[1].clear();
        std::vector<bbox_t> detections = detector->detect(images[1]);
        for (size_t i = 0; i < detections.size(); i++) {
            bbox_t d = detections[i];
            if (d.obj_id == 0) {
                //cv::rectangle(images[1], cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(255, 255, 255), 1);
                fbox f(d);
                frames[1].crops.push_back(frames[1].getCrop(images[1], &f));
                frames[1].detections.push_back(f);
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "section 2 duration: " << QString::number(duration.count()).toStdString() << std::endl;
}

void MainWindow::section3()
{
    auto start = std::chrono::high_resolution_clock::now();
    if (images[2].rows > 0) {
        featureModel->run(&frames[2]);

        DETECTIONS detections;
        frames[2].writeToDetections(&detections);

        ModelDetection::getInstance()->dataMoreConf((float)args_min_confidence, detections);
        ModelDetection::getInstance()->dataPreprocessing(args_nms_max_overlap, detections);

        mytracker->predict();
        mytracker->update(detections);

        std::vector<RESULT_DATA> result;
        for (Track& track : mytracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
        }

        for (size_t i = 0; i < result.size(); i++) {
            DETECTBOX box = result[i].second;
            cv::Rect rect = cv::Rect(box[0], box[1], box[2], box[3]);
            cv::rectangle(images[2], rect, cv::Scalar(255, 0, 0), 1);
            cv::putText(images[2], QString::number(result[i].first).toStdString(), cv::Point(rect.x, rect.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        lblImage->setPixmap(QPixmap::fromImage(QImage(images[2].data, images[2].cols, images[2].rows, QImage::Format_BGR888)));
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "section 3 duration: " << QString::number(duration.count()).toStdString() << std::endl;
}

MainWindow::~MainWindow()
{
}


/*
cv::Mat composite(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
for (size_t i = 0; i < frames[2].crops.size(); i++) {
    if (i < 5) {
        cv::Mat crop = frames[2].crops[i];
        crop.copyTo(composite(cv::Rect(i*64, 0, crop.cols, crop.rows)));
    }
}
lblImage->setPixmap(QPixmap::fromImage(QImage(composite.data, composite.cols, composite.rows, QImage::Format_BGR888)));
*/

/*
for (size_t i = 0; i < frames[2].detections.size(); i++) {
    DETECTION_ROW tmpRow;
    float x = frames[2].detections[i].x;
    float y = frames[2].detections[i].y;
    float w = frames[2].detections[i].w;
    float h = frames[2].detections[i].h;
    tmpRow.tlwh = DETECTBOX(x, y, w, h);
    tmpRow.confidence = frames[2].detections[i].confidence;
    for (int j = 0; j < feature_size; j++) {
        tmpRow.feature[j] = frames[2].features[i][j];
    }
    detections.push_back(tmpRow);
}
*/
