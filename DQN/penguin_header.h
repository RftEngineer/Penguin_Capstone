#ifndef PENGUIN_HEADER_H
#define PENGUIN_HEADER_H

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/thread/thread.hpp>


#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"

#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/data_transformer.hpp"
#include <Windows.h>
#include <wincodec.h>
#include <Winuser.h>

#include "boost/algorithm/string.hpp"
#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using caffe::SGDSolver;
using std::ostringstream;

/*typedef enum Action{
	Up = 0,
	Down = 1,
	Left = 2,
	Right = 3,
	Jump = 4
};
*/
enum Action{
	Nothing = 0,
	Up = 1,
	Down = 2,
	Left = 3,
	Right = 4,
	Jump = 5
};
#endif