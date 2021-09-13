// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.


#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gps/gps_helper.h"

using namespace std;
using namespace std::chrono;

constexpr char kInputStream[]  = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[]   = "Drowsiness Detector";
constexpr char kIris[]         = "face_landmarks_with_iris";
constexpr char kIrisPresence[] = "face_landmarks_with_iris_presence";

constexpr char FILE_UP_AND_RUNNING[]                 = "up_and_running.mp3";
constexpr char FILE_DROWSINESS_ALERT[]               = "drowsiness_alert.mp3";
constexpr char FILE_ALARM[]                          = "alarm.wav";
constexpr char FILE_GPS_SIGNAL_LOST[]                = "gps_signal_lost.mp3";
constexpr char FILE_GPS_SIGNAL_HAS_2D_FIX[]          = "gps_signal_has_2d_fix.mp3";
constexpr char FILE_GPS_SIGNAL_HAS_3D_FIX[]          = "gps_signal_has_3d_fix.mp3";
constexpr char FILE_GPS_SPEED_CANNOT_BE_DETERMINED[] = "gps_speed_cannot_be_determined.mp3";
constexpr char FILE_GPS_SPEED_IS_LOW[]               = "gps_speed_is_low.mp3";
constexpr char FILE_GPS_SPEED_EXCEEDS_THRESHOLD[]    = "gps_speed_exceeds_threshold.mp3";

constexpr char FILE_SYSTEM_LOST_GPS_SIGNAL[]         = "system_lost_gps_signal.mp3";
constexpr char FILE_SYSTEM_CAN_DETERMINE_SPEED[]     = "system_can_determine_speed.mp3";

const size_t FRAME_WIDTH = 640;
const size_t FRAME_HEIGHT = 480;
const size_t EYE_CONTOUR_POINTS = 16;

const int GPS_SPEED_IS_NAN            = -1;
const int GPS_SPEED_IS_LOW            =  0;
const int GPS_SPEED_EXCEEDS_THRESHOLD =  1;

// See all landmark indexes: https://github.com/google/mediapipe/blob/master/mediapipe/graphs/iris_tracking/calculators/update_face_landmarks_calculator.cc
// and vizualization: https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
const size_t left_eye_indexes[]  = { 33,  7,163,144,145,153,154,155,133,173,157,158,159,160,161,246};
const size_t right_eye_indexes[] = {263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466};

const size_t LEFT_EYE_LEFT_POINT    = 33;
const size_t LEFT_EYE_RIGHT_POINT   = 133;
const size_t LEFT_EYE_BOTTOM_POINT  = 145;
const size_t LEFT_EYE_TOP_POINT     = 159;

const size_t RIGHT_EYE_LEFT_POINT   = 362;
const size_t RIGHT_EYE_RIGHT_POINT  = 263;
const size_t RIGHT_EYE_BOTTOM_POINT = 374;
const size_t RIGHT_EYE_TOP_POINT    = 386;

const cv::Scalar WHITE_COLOR(255, 255, 255);
const cv::Scalar RED_COLOR  (  0,   0, 255);
const cv::Scalar GREEN_COLOR(  0, 255,   0);
const cv::Scalar BLUE_COLOR (255,   0,   0);

cv::VideoWriter writer;

double closed_ear = 0.30;
size_t drowsiness_interval = 2; // Timeout in seconds before the next drowsinness alert can be detected, default is 2 sec 
double volume = 0.20;  // volume of audio in range [0..1]
size_t MAX_FPS = 30;
size_t alarm_interval = 8; // Timeout in seconds before the next Alert signal, default is 8 sec
string output_video;  // "yes" - write video to a file, default is "no"
string display_video; // "yes" - show video on display, default is "no"
string flip_video; // "yes" - flip video horizontally, default is "no"
string use_gps; // "yes" - use GPS to get speed, default is "no"
string gps_logger; // "yes" - enable GPS logging into CSV file, default is "no"
string gps_logger_file; // file template for GPS logger (%T is timestamp %Y%m%d_%H%M%S, e.g. 20210831_140102)
size_t gps_logger_interval; // Timeout in seconds before the next GPS logging, default is 2 sec
string gps_audio; // "yes" - play audio files when GPS speed status changes (cannot be determined, is low, exceeds threshold), default is "no"
size_t output_video_duration = 300; // default duration of video file
string output_video_path; // file template for video output (%T is timestamp %Y%m%d_%H%M%S, e.g. 20210831_140102)
string video_file_name;
auto video_file_start_time = high_resolution_clock::now();
size_t GPS_SPEED_THRESHOLD = 5;

int gps_speed_status = GPS_SPEED_IS_NAN;

int gps_status = STATUS_NO_FIX;
int gps_fix_mode = MODE_NOT_SEEN;
 
mutex m;
condition_variable cond;
bool ready = false;
bool processed = true;
bool keep_running = true;
cv::Mat* frame_ptr = nullptr;
mutex play_audio_mutex;

bool grab_frames = true;
bool display_video_flag = false;
bool output_video_flag = false;
bool flip_video_flag = false;
bool use_gps_flag = false;
bool gps_logger_flag = false;
bool alerts_enabled = false;
bool gps_audio_flag = false;

ABSL_FLAG(std::string, calculator_graph_config_file, "",    
		"Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
	      "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");
ABSL_FLAG(size_t, output_video_duration, 300,
                "Duration of output video file in seconds. Default is 300 sec (5 mins).");
ABSL_FLAG(double, closed_eyes_aspect_ratio, 0.30,
                "Aspect Ratio for closed eyes. Default is 0.30.");
ABSL_FLAG(size_t, drowsiness_interval, 2,
                "Drowsiness Interval in seconds Default is 2 sec.");
ABSL_FLAG(double, volume, 0.20,
                "Volume of audio in range [0..1]. Default is 0.20.");
ABSL_FLAG(size_t, max_fps, 30,
                "Maximum Frames Per Second. Default is 30.");
ABSL_FLAG(size_t, alarm_interval, 8,
                "Interval between alarm sound in seconds.. Default is 8 sec.");
ABSL_FLAG(std::string, output_video, "no",
              "\"yes\" - write video to a file, default is \"no\"");
ABSL_FLAG(std::string, display_video, "no",
              "\"yes\" - show video on display, default is \"no\"");
ABSL_FLAG(std::string, flip_video, "no",
              "\"yes\" - flip video horizontally, default is \"no\"");
ABSL_FLAG(std::string, use_gps, "no",
              "\"yes\" - use GPS to get speed, default is \"no\"");
ABSL_FLAG(size_t, gps_speed_threshold, 5,
              "Sensitivity threshold of GPS speed, default is 5 MPH");
ABSL_FLAG(std::string, gps_logger, "no",
              "\"yes\" - enable GPS logging into CSV file, default is \"no\"");
ABSL_FLAG(std::string, gps_logger_file, "",
              "File template for GPS logger (%T is timestamp %Y%m%d_%H%M%S, e.g. 20210831_140102)");
ABSL_FLAG(size_t, gps_logger_interval, 2,
              "Timeout in seconds before the next GPS logging, default is 2 sec");
ABSL_FLAG(std::string, gps_audio, "no",
              "\"yes\" - play audio files when GPS speed status changes (cannot be determined, is low, exceeds threshold), default is \"no\"");

inline bool file_exists (const string& name) {
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}

int backup_file(const string& file_name) {
	int rc = 0;
	// keep last 9 copies of file with the same name:
	// file_name.ext, file_name.1.ext, file_name.2.ext, ..., filename.9.ext
	auto pos = file_name.find_last_of('.');
	string name = file_name.substr(0,pos);
	string ext  = file_name.substr(pos+1);
	// last 9th file will be deleted if exists
        string last_copy = name+".9."+ext;	
	if( file_exists(last_copy) ) rc = remove(last_copy.c_str());
	if(rc) return rc;
	// the rest of copies should be renamed: name.<i>.ext --> name.<i+1>.ext
	for(size_t i=8; i>0; i--) {
		string existing_file = name+"."+to_string(i)+"."+ext;
		string renamed_file  = name+"."+to_string(i+1)+"."+ext;
		if( file_exists(existing_file) ) rc = rename(existing_file.c_str(), renamed_file.c_str());
        	if(rc) return rc;
	}
	// last step - rename current file to .1.
	string first_copy = name+".1."+ext;
	rc = rename(file_name.c_str(), first_copy.c_str());
	return rc;
}	

int video_writer() {

    int rc = 0;

    while(keep_running && !rc) {
        
	unique_lock<mutex> lk(m);
        cond.wait(lk, []{return ready;});

        if(frame_ptr) {
		if (writer.isOpened()) {
			auto time_now = high_resolution_clock::now();
		        microseconds duration = duration_cast<microseconds>(time_now - video_file_start_time);
		        if(duration.count() >= output_video_duration * 1e6) {	
				writer.release();
				LOG(INFO) << "Video file " << video_file_name << " closed";
			}
		}
		if (!writer.isOpened()) {
			video_file_name = output_video_path;
			auto pos = video_file_name.find("%T");
			if(pos != string::npos){
				char time_str[100];
				time_t rawtime;
				struct tm * timeinfo;
				time (&rawtime);
				timeinfo = localtime (&rawtime);
				strftime (time_str,sizeof(time_str),"%Y%m%d_%H%M%S",timeinfo);
				video_file_name = video_file_name.replace(pos, 2, string(time_str));
				if( file_exists(video_file_name) ) rc = backup_file(video_file_name);
				if(rc) {
					LOG(INFO)<<"Error while backing up video file";
					break;
				}
			}
        		LOG(INFO) << "Writing video to file " << video_file_name;
			video_file_start_time = high_resolution_clock::now();
        		writer.open(video_file_name,
                    		cv::CAP_GSTREAMER,
                    		mediapipe::fourcc('x', '2', '6', '4'),
                    		MAX_FPS, 
                    		frame_ptr->size());
        		if(!writer.isOpened()) {
				LOG(INFO) << "Can't open video stream for writing";
				rc = -1;
				break;
			}
      		}
		writer.write(*frame_ptr);	    
		delete frame_ptr;
		frame_ptr = nullptr;
        }

        processed = true;
        ready = false;

        lk.unlock();
        cond.notify_one();
    }
    return rc;
}

void transcode_landmark(const mediapipe::NormalizedLandmarkList* all_landmarks,
		        vector<cv::Point>* landmark, 
   		        const size_t* indexes) {
  for(int i=0; i<EYE_CONTOUR_POINTS; i++) {
    (*landmark)[i] = cv::Point(round(all_landmarks->landmark(indexes[i]).x()*FRAME_WIDTH), round(all_landmarks->landmark(indexes[i]).y()*FRAME_HEIGHT));
  }
}

// Function calculates distance between 2 points in 3D dimension
double distance(const mediapipe::NormalizedLandmark* p1, const mediapipe::NormalizedLandmark* p2) {
	return sqrt( pow(p1->x()-p2->x(),2) + pow(p1->y()-p2->y(),2) + pow(p1->z()-p2->z(),2) );
}

// Function calculates Eye Aspect Ration as ration of distances between top-bottom and left-right points
double ear(const mediapipe::NormalizedLandmarkList* landmarks) {
	double left_ear  = distance(&landmarks->landmark(LEFT_EYE_TOP_POINT),   &landmarks->landmark(LEFT_EYE_BOTTOM_POINT)) / 
	                   distance(&landmarks->landmark(LEFT_EYE_LEFT_POINT),  &landmarks->landmark(LEFT_EYE_RIGHT_POINT));
	double right_ear = distance(&landmarks->landmark(RIGHT_EYE_TOP_POINT),  &landmarks->landmark(RIGHT_EYE_BOTTOM_POINT)) /
                           distance(&landmarks->landmark(RIGHT_EYE_LEFT_POINT), &landmarks->landmark(RIGHT_EYE_RIGHT_POINT));
	return (left_ear+right_ear)/2.0;
}

void play_audio(const char* file_name) {
    unique_lock<mutex> lk(play_audio_mutex);
    char str[128];
    snprintf(str, sizeof(str), "gst-play-1.0 --volume=%4.2f %s", volume, file_name);
    system(str);
}

void play_alarm() {
	play_audio(FILE_DROWSINESS_ALERT);
	play_audio(FILE_ALARM);
}

void log_func(string msg) {
    LOG(INFO) << msg;
}

int read_command_line_flags() {
  
  closed_ear = absl::GetFlag(FLAGS_closed_eyes_aspect_ratio);
  LOG(INFO) << "Closed Eyes Aspect Ratio: " << closed_ear;

  drowsiness_interval = absl::GetFlag(FLAGS_drowsiness_interval);
  LOG(INFO) << "Drowsiness Interval: " << drowsiness_interval << " seconds";

  volume = absl::GetFlag(FLAGS_volume);
  LOG(INFO) << "Volume: " << volume;

  MAX_FPS = absl::GetFlag(FLAGS_max_fps);
  LOG(INFO) << "MAX_FPS: " << MAX_FPS;

  alarm_interval = absl::GetFlag(FLAGS_alarm_interval);
  LOG(INFO) << "Alarm Interval: " << alarm_interval << " seconds";

  display_video = string(absl::GetFlag(FLAGS_display_video));
  LOG(INFO) << "Display Video: " << display_video;

  output_video = string(absl::GetFlag(FLAGS_output_video));
  LOG(INFO) << "Output Video: " << output_video;

  output_video_path = string(absl::GetFlag(FLAGS_output_video_path));
  LOG(INFO) << "Output Video Path: " << output_video_path;

  output_video_duration = absl::GetFlag(FLAGS_output_video_duration);
  LOG(INFO) << "Output Video Duration: " << output_video_duration << " seconds";

  flip_video = string(absl::GetFlag(FLAGS_flip_video));
  LOG(INFO) << "Flip Video: " << flip_video;

  use_gps = string(absl::GetFlag(FLAGS_use_gps));
  LOG(INFO) << "Use GPS: " << use_gps;

  GPS_SPEED_THRESHOLD = absl::GetFlag(FLAGS_gps_speed_threshold);
  LOG(INFO) << "GPS Speed Threshold: " << GPS_SPEED_THRESHOLD << " mph";

  gps_logger = string(absl::GetFlag(FLAGS_gps_logger));
  LOG(INFO) << "GPS Logger: " << gps_logger;

  gps_logger_file = string(absl::GetFlag(FLAGS_gps_logger_file));
  LOG(INFO) << "GPS Logger File: " << gps_logger_file;

  gps_logger_interval = absl::GetFlag(FLAGS_gps_logger_interval);
  LOG(INFO) << "GPS Logger Interval: " << gps_logger_interval << " seconds";

  gps_audio = string(absl::GetFlag(FLAGS_gps_audio));
  LOG(INFO) << "GPS Audio: " << gps_audio;

  if(!strcasecmp(display_video.c_str(), "yes")) display_video_flag = true;
  if(!strcasecmp(output_video.c_str(),  "yes")) output_video_flag  = true;
  if(!strcasecmp(flip_video.c_str(),    "yes")) flip_video_flag    = true;
  if(!strcasecmp(use_gps.c_str(),       "yes")) use_gps_flag       = true;
  if(!strcasecmp(gps_logger.c_str(),    "yes")) gps_logger_flag    = true;
  if(!strcasecmp(gps_audio.c_str(),     "yes")) gps_audio_flag     = true;
 
  if(!use_gps_flag) alerts_enabled = true; // enable alerts if GPS speed detecting is not used 
}

absl::Status RunMPPGraph() {

  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  LOG(INFO) << "GPU:1";
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  LOG(INFO) << "GPU:2";
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0, cv::CAP_GSTREAMER);
  }
  RET_CHECK(capture.isOpened());

  unique_ptr<thread> video_writer_thread(nullptr);

  if(output_video_flag) {
  	video_writer_thread = make_unique<thread>(video_writer);
  	LOG(INFO) << "Video writer thread started";
  }
    
  capture.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
  // don't set FPS for camera explicitely, use default value
  //capture.set(cv::CAP_PROP_FPS, MAX_FPS);

  bool eyes_were_closed = false;
  auto eyes_closing_time = high_resolution_clock::now();
  auto alarm_time = high_resolution_clock::now();

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller iris_poller,
                   graph.AddOutputStreamPoller(kIris));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller iris_presence_poller,
                   graph.AddOutputStreamPoller(kIrisPresence));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  size_t fps = 0;

  future<void> rt = async(launch::async, play_audio, FILE_UP_AND_RUNNING);

  future<void> alarm_return;
  future<void> gps_speed_return;

  gps_data_t* gps_data_ptr = nullptr;

  LOG(INFO) << "Start grabbing and processing frames.";

  while (grab_frames) {
    //
    // Take a start time for a frame processing
    auto frame_start_time = high_resolution_clock::now();
    //
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGBA, 4);
    if (!load_video) {
      if(flip_video_flag) cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return absl::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> absl::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
              gpu_frame.format(), 0, gpu_helper.GetGlVersion());
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return absl::OkStatus();
        }));

    cv::Mat output_frame_mat; 

    if(display_video_flag || output_video_flag) {
        // Convert back to opencv for display or saving.
        output_frame_mat = mediapipe::formats::MatView(output_frame.get());
        if (output_frame_mat.channels() == 4)
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
        else
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    }

    if(use_gps_flag) {
        gps_data_ptr = gps_helper_read();
        if(gps_data_ptr) {
            if(gps_audio_flag) {
                if( gps_status == STATUS_NO_FIX
                 && gps_fix_mode <= MODE_NO_FIX
                 && gps_data_ptr->status > STATUS_NO_FIX 
                 && gps_data_ptr->fix.mode > MODE_NO_FIX ) {
                    LOG(INFO) << "Notification: system can determine speed now, using GPS!";
                    async(launch::async, play_audio, FILE_SYSTEM_CAN_DETERMINE_SPEED);
                }
                else if(gps_status > STATUS_NO_FIX
                 && gps_fix_mode > MODE_NO_FIX
                 && gps_data_ptr->status == STATUS_NO_FIX
                 && gps_data_ptr->fix.mode <= MODE_NO_FIX ) {
                    LOG(INFO) << "Warning: system lost GPS signal!";
                    async(launch::async, play_audio, FILE_SYSTEM_LOST_GPS_SIGNAL);
                }
            }
            gps_status = gps_data_ptr->status;
            gps_fix_mode = gps_data_ptr->fix.mode;

            double mph = round(gps_data_ptr->fix.speed*MPS_TO_MPH);
            char mph_str[100];
            snprintf(mph_str, sizeof(mph_str), "%3.0f", mph);
            if(display_video_flag || output_video_flag) {
                char mph_on_frame[100];
                snprintf(mph_on_frame, sizeof(mph_on_frame), "MPH:%3.0f", mph);
                cv::putText(output_frame_mat, mph_on_frame, cv::Point(15, 220), cv::FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR);
            }
            if(isnan(mph) && gps_speed_status!=GPS_SPEED_IS_NAN) {
                gps_speed_status = GPS_SPEED_IS_NAN;
                alerts_enabled = false;
                LOG(INFO) << "Alerts disabled because GPS speed cannot be determined";
                //if(gps_audio_flag) gps_speed_return = async(launch::async, play_audio, FILE_GPS_SPEED_CANNOT_BE_DETERMINED);
            }else if(mph<=GPS_SPEED_THRESHOLD && gps_speed_status!=GPS_SPEED_IS_LOW) {
                gps_speed_status = GPS_SPEED_IS_LOW;
                alerts_enabled = false;
                LOG(INFO) << "Alerts disabled because GPS speed is low";
                //if(gps_audio_flag) gps_speed_return = async(launch::async, play_audio, FILE_GPS_SPEED_IS_LOW);
            }else if(mph>GPS_SPEED_THRESHOLD && gps_speed_status!=GPS_SPEED_EXCEEDS_THRESHOLD){
                gps_speed_status = GPS_SPEED_EXCEEDS_THRESHOLD;
                alerts_enabled = true;
                LOG(INFO) << "Alerts enabled because GPS speed " << mph_str << "mph exceeds threshold " << GPS_SPEED_THRESHOLD << "mph";
                //if(gps_audio_flag) gps_speed_return = async(launch::async, play_audio, FILE_GPS_SPEED_EXCEEDS_THRESHOLD);
            }else {
                //LOG(INFO) << "Unhandled transition of speed= " << mph_str << " and GPS_SPEED_STATUS=" << gps_speed_status;
            }
        }
    }


    bool eyes_closed = false;

    // Process Iris Landmark
    mediapipe::Packet iris_presence_packet;
    if (iris_presence_poller.Next(&iris_presence_packet)) {
      auto is_iris_present = iris_presence_packet.Get<bool>();
      if (is_iris_present) {
        mediapipe::Packet iris_packet;
        if (iris_poller.Next(&iris_packet)) {
            //LOG(INFO) << "Iris packet detected";
	        auto &output_landmarks = iris_packet.Get < mediapipe::NormalizedLandmarkList > ();
            auto ear_value = ear(&output_landmarks);
            if(ear_value < closed_ear) eyes_closed = true;
            if(display_video_flag || output_video_flag) {
                vector< vector<cv::Point> > contours(2);
	            auto left_eye  = vector<cv::Point> (EYE_CONTOUR_POINTS);
	            auto right_eye = vector<cv::Point> (EYE_CONTOUR_POINTS);
	            transcode_landmark(&output_landmarks, &left_eye, left_eye_indexes);
                transcode_landmark(&output_landmarks, &right_eye, right_eye_indexes);
                contours[0] = left_eye;
	            contours[1] = right_eye;
	            cv::drawContours( output_frame_mat, contours, 0, WHITE_COLOR);
	            cv::drawContours( output_frame_mat, contours, 1, WHITE_COLOR);
	            //
  	            char ear_str[100];
                snprintf(ear_str, sizeof(ear_str), "EAR: %4.2f", ear_value);
                cv::putText(output_frame_mat, ear_str, cv::Point(FRAME_WIDTH-90, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR);
            }
	    }
      }
    }
    if(eyes_closed) {
	    if(eyes_were_closed) {
    		auto time_now = high_resolution_clock::now();
    		microseconds duration = duration_cast<microseconds>(time_now - eyes_closing_time);
    		if(duration.count() > drowsiness_interval * 1e6) {
                if(alerts_enabled && (display_video_flag || output_video_flag)) 
                    cv::putText(output_frame_mat, "DROWSINESS ALERT", cv::Point(200, 30), cv::FONT_HERSHEY_DUPLEX, 0.8, RED_COLOR, 2);
    			microseconds alarm_duration = duration_cast<microseconds>(time_now - alarm_time);
    			if(alarm_duration.count() > alarm_interval * 1e6) {			
                    alarm_time = high_resolution_clock::now();	
                    if(alerts_enabled) {
    	    			LOG(INFO) << "DROWSINESS ALERT";
	    		    	alarm_return = async(launch::async, play_alarm);
                    }
    			}
	    	}
    	} else {
	    	eyes_were_closed = true;
		    eyes_closing_time = high_resolution_clock::now(); 
    	}
    } else {
	    eyes_were_closed = false;
    }
    if(display_video_flag || output_video_flag) { 
        // FPS calculation is based on processing time of the previous frame
        char fps_str[100];
        snprintf(fps_str, sizeof(fps_str), "FPS: %d", fps);
        cv::putText(output_frame_mat, fps_str, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR);
        // Display current time in the bottom right corner
        char time_str[100];
        time_t rawtime;
        struct tm * timeinfo;
        time (&rawtime);
        timeinfo = localtime (&rawtime);
        strftime (time_str,sizeof(time_str),"%c",timeinfo);
        cv::putText(output_frame_mat, time_str, cv::Point(FRAME_WIDTH-280, FRAME_HEIGHT-10), cv::FONT_HERSHEY_SIMPLEX, 0.45, GREEN_COLOR);
    }
    if (output_video_flag) {
      {     // need to wait for Video writer thread to finish writing previous frame
            unique_lock<mutex> lk(m);
            cond.wait(lk, []{return processed;});
            processed = false;
	        // prepare copy of new frame to be written to the file
            ready = true;
            frame_ptr = new cv::Mat(output_frame_mat.clone());
      }
      cond.notify_one();
    }
    if(display_video_flag) cv::imshow(kWindowName, output_frame_mat);
    // Press Q/q key to exit.
    const int pressed_key = cv::waitKey(1);
    if (pressed_key==113/*q*/ || pressed_key==81/*Q*/) grab_frames = false;
    // Calculate frame diration in microseconds and FPS
    auto frame_stop_time = high_resolution_clock::now(); 
    microseconds duration = duration_cast<microseconds>(frame_stop_time - frame_start_time);
    // If frame processed faster than 1e6/MAX_FPS then we can sleep the rest of the time
    if(duration.count() < 1.0e6/MAX_FPS-100.0) {
	    microseconds sleep_time(long(1.0e6/MAX_FPS - duration.count() - 100.0));
	    this_thread::sleep_for(sleep_time);
	    //LOG(INFO) << "Slept for " << sleep_time.count();
	    // Recalculate duration
	    frame_stop_time = high_resolution_clock::now();
        duration = duration_cast<microseconds>(frame_stop_time - frame_start_time);
    } 
    //else {
    //	    LOG(INFO) << "Didn\'t sleep";
    //}
    fps = round(1.0e6/duration.count());
    //LOG(INFO) << "FPS: " << round(1.0e6/duration.count());
  }
  LOG(INFO) << "Main loop terminated";
  
  if(output_video_flag) {
	{
		unique_lock<mutex> lk(m);
 		cond.wait(lk, []{return processed;});
		keep_running = false;
		ready = true;
	}
  	cond.notify_one();	
  	video_writer_thread->join();
  	LOG(INFO) << "Video writer thread terminated";
  }

  cv::destroyAllWindows();
  LOG(INFO) << "All Windows destoyed";

  LOG(INFO) << "Shutting down";
  if (writer.isOpened()) {
	writer.release();
	LOG(INFO) << "Video writer closed";
  }
  if (capture.isOpened()) {
	capture.release();
	LOG(INFO) << "Video capture closed";
  }
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));

  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {

  int rc = 0;
  
  google::InitGoogleLogging(argv[0]);
  //gflags::ParseCommandLineFlags(&argc, &argv, true);
  absl::ParseCommandLine(argc, argv);
  
  read_command_line_flags();
  
  if(use_gps_flag) {
        gps_helper_register_logger_callback(log_func);
        rc = gps_helper_open();
        if(rc) return rc;
  }
 
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
 
  if(use_gps_flag) {
        rc = gps_helper_close();
        if(rc) return rc;
  }

  this_thread::sleep_for(milliseconds(2000));
  LOG(INFO) << "Exit";

  return EXIT_SUCCESS;
}
