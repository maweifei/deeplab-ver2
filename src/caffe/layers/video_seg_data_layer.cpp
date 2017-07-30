#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_seg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
VideoSegDataLayer<Dtype>::~VideoSegDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    string bgfn = "";
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
    }
    iss >> bgfn;
    lines_.push_back(std::make_pair(imgfn, segfn+" " + bgfn));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  //we concate background into image channels
  //const int channels = cv_img.channels();
  const int channels = cv_img.channels()*2;
  LOG(INFO) << "Image and background channels is " << channels;
  
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  //const int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_width = 0;
  int crop_height = 0;
  CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
	|| (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (transform_param.has_crop_size()) {
    crop_width = transform_param.crop_size();
    crop_height = transform_param.crop_size();
  } 
  if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
    crop_width = transform_param.crop_width();
    crop_height = transform_param.crop_height();
  }

  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_width > 0 && crop_height > 0) {
    top[0]->Reshape(batch_size, channels, crop_height, crop_width);
    this->transformed_data_.Reshape(batch_size, channels, crop_height, crop_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, crop_height, crop_width);
    }

    //label
    top[1]->Reshape(batch_size, 1, crop_height, crop_width);
    this->transformed_label_.Reshape(batch_size, 1, crop_height, crop_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, crop_height, crop_width);
    }
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(batch_size, channels, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
    }

    //label
    top[1]->Reshape(batch_size, 1, height, width);
    this->transformed_label_.Reshape(batch_size, 1, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
    }
  }
  // image dimensions, for each image, stores (img_height, img_width)
  top[2]->Reshape(batch_size, 1, 1, 2);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].dim_.Reshape(batch_size, 1, 1, 2);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
  // image_dim
  LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
	    << top[2]->channels() << "," << top[2]->height() << ","
	    << top[2]->width();
}

template <typename Dtype>
void VideoSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoSegDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data     = batch->data_.mutable_cpu_data();
  Dtype* top_label    = batch->label_.mutable_cpu_data(); 
  Dtype* top_data_dim = batch->dim_.mutable_cpu_data();

  const int max_height = batch->data_.height();
  const int max_width  = batch->data_.width();

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  const int lines_size = lines_.size();
  int top_data_dim_offset;

  const int crop_height = batch->data_.height();
  const int crop_width = batch->data_.width();
  //const int bg_channels = 3;
  const int bg_channels_offset = 3;
  //TransformationParameter transform_param = this->layer_param_.transform_param();
  //vector<Dtype> mean_values_ = transform_param.mean_value();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    top_data_dim_offset = batch->dim_.offset(item_id);

    std::vector<cv::Mat> cv_img_seg;

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    int img_row, img_col;
    cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].first,
	  new_height, new_width, is_color, &img_row, &img_col));

    // TODO(jay): implement resize in ReadImageToCVMat
    // NOTE data_dim may not work when min_scale and max_scale != 1
    top_data_dim[top_data_dim_offset]     = static_cast<Dtype>(std::min(max_height, img_row));
    top_data_dim[top_data_dim_offset + 1] = static_cast<Dtype>(std::min(max_width, img_col));

    if (!cv_img_seg[0].data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
    }
    
    string label_bg_line_ = lines_[lines_id_].second;
    //LOG(INFO) << "label bg line " << label_bg_line_;
    std::istringstream iss(label_bg_line_);
    string label_file = "";
    
    if (label_type == ImageDataParameter_LabelType_PIXEL) {
      iss >> label_file;
      cv_img_seg.push_back(ReadImageToCVMat(root_folder + label_file,
					    new_height, new_width, false));
      if (!cv_img_seg[1].data) {
	DLOG(INFO) << "Fail to load seg: " << root_folder + label_file;
      }
    }
    else if (label_type == ImageDataParameter_LabelType_IMAGE) {
      const int label = atoi(lines_[lines_id_].second.c_str());
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(seg);      
    }
    else {
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(seg);
    }



    //load background image
    string bg_file= "";
    iss >> bg_file;
    //LOG(INFO) << "bg line " << bg_file;
    //LOG(INFO) << "new height " << new_height << " new width " << new_width;
    //LOG(INFO) << "crop height " << crop_height << " crop width " << crop_width;
    cv::Mat bg_img = ReadImageToCVMat(root_folder + bg_file,
	                    crop_height, crop_width, is_color, &img_row, &img_col);
    //bg_img.convertTo(bg_img,CV_64F);
    cv_img_seg.push_back(bg_img);


    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset;
    offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);

    this->data_transformer_->TransformImgAndSeg_video(cv_img_seg,
	 &(this->transformed_data_), &(this->transformed_label_),
	 ignore_label,bg_channels_offset);

    
    //Dtype* transformed_data = this->transformed_data_.mutable_cpu_data();
    //LOG(INFO) << "batch size" << this->transformed_data_.num() << "transformed height " << this->transformed_data_.height() << " transformed width " << this->transformed_data_.width() << "transformed channels" << this->transformed_data_.channels();
    //const double* bg_ptr;
    //const uchar* bg_ptr;
    //int top_index;
    //transformed_dat 
    /*
    for (int h = 0; h < crop_height; ++h) {
        bg_ptr = bg_img.ptr<double>(h);
        //bg_ptr = bg_img.ptr<uchar>(h);
        
        int bg_index = 0;

        for (int w = 0; w < crop_width; ++w) {
            for (int c = 0; c < bg_channels; ++c) {
                top_index = ((c+bg_channels_offset) * crop_height + h) * crop_width + w;
                //top_index = (c * crop_height + h) * crop_width + w;
                CHECK_LE(top_index, 6*crop_height*crop_width);
                Dtype pixel = static_cast<Dtype>(bg_ptr[bg_index++]);
                //top_index = 0;
                //LOG(INFO) << top_index << " " << 6*crop_height*crop_width;
                transformed_data[top_index] = (pixel - mean_values_[c]);
                 
            }
        }
    }
    */
    

    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoSegDataLayer);
REGISTER_LAYER_CLASS(VideoSegData);

}  // namespace caffe
