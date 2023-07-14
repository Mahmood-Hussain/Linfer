


#include <fstream>
#include <string>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "apps/yolov5/yolov5.hpp"
#include "trt_common/ilogger.hpp"

using namespace std;

const char* cocolabels[] = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
};

bool exists(const string& path){
    return access(path.c_str(), R_OK) == 0;
}

int main(){
    cv::Mat image = cv::imread("rq.jpg");
    auto yolov5 = Yolo::create_infer("yolov5s.trt", Yolo::Type::V5);
    auto boxes = yolov5->commit(image).get();
    INFO("5555555555555555555555");
    for(auto& box : boxes){
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

        auto name = cocolabels[box.label];
        auto caption = cv::format("%s %.2f", name, box.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
        cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("image-draw2.jpg", image);
    INFO("999999999999999999");
    return 0;
}

