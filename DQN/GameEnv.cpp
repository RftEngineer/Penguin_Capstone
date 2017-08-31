#include "GameEnv.h"
#include "penguin_header.h"
#include <memory>
#include <vector>
#include <iostream>
namespace dqn{

		PenguinEnv::PenguinEnv  (int frameSize) {
			for (int i = 0; i < 5; i++)
				legal_actions_.push_back(static_cast<ActionCode>(i));
			gameover = false;
			spd_is_zero = false;
			screenmode = frameSize;
			reward = 0;

		}
		dqn::PenguinEnv::FrameDataSp PenguinEnv::PreprocessScreen(cv::Mat T1)
		{
			//cv::Mat T2;
			//T2.create(croppedFrameSize, croppedFrameSize, CV_8UC1);
			//cv::imwrite("C:/Capstone/temp.jpg", T1);
			//LOG(INFO) << "Save Complete";
			cv::resize(T1, T1, cv::Size(dqn::PenguinEnv::croppedFrameSize, dqn::PenguinEnv::croppedFrameSize), 0, 0, CV_INTER_NN);
			//LOG(INFO) << "Resize complete";

			cv::cvtColor(T1, T1, CV_RGB2GRAY);
			//LOG(INFO) << "Grayscale complete";

			auto screen = std::make_shared<PenguinEnv::FrameData>();
			for (auto i = 0; i < PenguinEnv::croppedFrameSize; ++i)
			{
				for (auto j = 0; j < PenguinEnv::croppedFrameSize; ++j)
				{
					(*screen)[i*PenguinEnv::croppedFrameSize + j] = T1.at<uint8_t>(i, j);
					//T2.at<uint8_t>(i, j) = (*screen)[i*croppedFrameSize + j];
				}
			}
			//LOG(INFO) << "Preprocessing done";
			//cv::imwrite("C:/Capstone/temp.jpg", T2);
			//LOG(INFO) << "Save Complete";
			return screen;
		}

		double PenguinEnv::ActNoop(int time) {
			//double reward = 0;
			keyInput(0);
			//reward++;
			double reward_ = double(time);
			return reward_;
		}

		double PenguinEnv::Act(int action, int time) {
			double reward_ = double(time);
			keyInput(action);
			//reward++;
			
			return reward_;
		}

		void PenguinEnv::keyInput(int keyNum)
		{
			//Sleep(20);
			//case 2://Down
			//	//keyReset();
			//	keybd_event(VK_UP, 0, KEYEVENTF_KEYUP, 0);
			//	Sleep(20);
			//	keybd_event(VK_DOWN, 0, 0, 0);
			//	break;
			//keyNum %= 5;
			//LOG(INFO) << "KeyInput occur! " << keyNum;
			switch (keyNum)
			{
			case 0://Nothing
			//	keyReset();
				break;
			case 1://Up
				//keyReset();
				keybd_event(VK_DOWN, 0, KEYEVENTF_KEYUP, 0);
				Sleep(15);
				keybd_event(VK_UP, 0, 0, 0);
				//cv::waitKey
				break;
			case 2://Left
				//keyReset();
				keybd_event(VK_RIGHT, 0, KEYEVENTF_KEYUP, 0);
				Sleep(15);
				keybd_event(VK_LEFT, 0, 0, 0);
				break;
			case 3://Right
				//keyReset();
				keybd_event(VK_LEFT, 0, KEYEVENTF_KEYUP, 0);
				Sleep(15);
				keybd_event(VK_RIGHT, 0, 0, 0);
				break;
			case 4:
				//keyReset();
				keybd_event(88, 0, KEYEVENTF_KEYUP, 0);
				Sleep(15);
				keybd_event(88, 0, 0, 0);
				break;
			}
			//keybd_event(keyNum, 0, 0, 0);
		}
		void PenguinEnv::keyReset()
		{
			//LOG(INFO) << "KeyReset";
			Sleep(15);
			keybd_event(VK_LEFT, 0, KEYEVENTF_KEYUP, 0);
			keybd_event(VK_RIGHT, 0, KEYEVENTF_KEYUP, 0);
			keybd_event(VK_UP, 0, KEYEVENTF_KEYUP, 0);
			keybd_event(VK_DOWN, 0, KEYEVENTF_KEYUP, 0);
			keybd_event(88, 0, KEYEVENTF_KEYUP, 0); //X
			//Reset();
		}

		const PenguinEnv::ActionVec& PenguinEnv::GetMinimalActionSet(){
			return legal_actions_;
		}

		void PenguinEnv::Reset()
		{
			//LOG(INFO) << "Calling Reset";
			
			keybd_event(76, 0, 0, 0);
			Sleep(100);
			keybd_event(76, 0, KEYEVENTF_KEYUP, 0);
			reward = 0;
			spd_is_zero = false;
			gameover = false;
			//Load savedata
		}
		cv::Mat PenguinEnv::setRoi(cv::Mat img_org, int minX, int minY, int maxX, int maxY)
		{
			//cv::Mat spdSrc = cv::imread("C:/Capstone/pic/spd.bmp");
			//cv::cvtColor(spdSrc, spdSrc, CV_BGR2GRAY);
			//cv::Mat overSrc = cv::imread("C:/Capstone/pic/gameover.bmp");
			//cv::cvtColor(overSrc, overSrc, CV_BGR2GRAY);
			//cv::Mat T3 = setRoi(T2, 186, 14, 196, 22);
			//cv::Mat T3 = setRoi(T2, 83, 61, 91, 69);
			return img_org(cv::Rect(cv::Point(minX, minY), cv::Point(maxX, maxY))); //설정된 roi 리턴
		}
		bool PenguinEnv::checkImg(cv::Mat org_img,  // Image for checking, already segmented.
			cv::Mat tpl_img,  // Image for match
			cv::Mat original) // Original Image
		{
			double minval, maxval;
			cv::Point minloc, maxloc;

			cv::matchTemplate(org_img, tpl_img, original, CV_TM_CCORR_NORMED);
			cv::threshold(original, original, 0.8, 1., CV_THRESH_TOZERO);

			cv::minMaxLoc(original, &minval, &maxval, &minloc, &maxloc);

			if (maxval >= 1)
			{
				return true;
			}
			else
				return false;
		}
		bool PenguinEnv::EpisodeOver(){
			//cv::Mat temp = setRoi(dqn::gameover, 89, 144, 97, 153);
			return gameover;

		};

		void PenguinEnv::setEpisodeOver(bool t1)
		{
			gameover = t1;
		}
		std::string PenguinEnv::action_to_string(PenguinEnv::ActionCode a){
			return PenguinEnv::m_action_to_string(static_cast<Action>(a));
		}
				
	dqn::PenguinEnvSp dqn::CreateEnvironment(int frameSize){
		return std::make_shared<PenguinEnv>(frameSize);
	}
}