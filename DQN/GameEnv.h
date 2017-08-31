#ifndef SRC_GAMEENV_H_
#define SRC_GAMEENV_H_

#include "penguin_header.h"

namespace dqn {
	class PenguinEnv;
	typedef std::shared_ptr<PenguinEnv> PenguinEnvSp;

	class PenguinEnv{
	public:
		typedef int ActionCode;
		typedef std::vector<int> ActionVec;
		PenguinEnv(int frameSize);
		static const auto rawFrameHeight = 212;
		static const auto rawFrameWidth = 249;
		static const auto croppedFrameSize = 84;
		static const auto croppedFrameDataSize = 7056; // 84 * 84
		static const auto inputFrameCount = 15;
		static const auto inputDataSize = croppedFrameDataSize * inputFrameCount;

		using FrameData = std::array<uint8_t, croppedFrameDataSize>;
		using FrameDataSp = std::shared_ptr<FrameData>;
		using State = std::array<FrameDataSp, inputFrameCount>;
		
		FrameDataSp PreprocessScreen(cv::Mat t1);

		cv::Mat setRoi(cv::Mat img_org, int minX, int minY, int maxX, int maxY);

		bool checkImg(cv::Mat org_img,  // Image for checking, already segmented.
			cv::Mat tpl_img,  // Image for match
			cv::Mat original); // Original Image

		double ActNoop(int time);

		double Act(int action,int time);

		void keyReset();

		void Reset();

		void keyInput(int keyNum);

		bool EpisodeOver();

		void setEpisodeOver(bool t1);

		static std::string m_action_to_string(Action a){
			std::string tmp_action_to_string[] = {
				"Nothing",
				"Up",
				"Down",
				"Left",
				"Right",
				"Jump"
			};
			return tmp_action_to_string[a];
		}

		std::string action_to_string(ActionCode a);

		const ActionVec& GetMinimalActionSet();
		
	private:
		ActionVec legal_actions_;
		int screenmode;
		bool gameover;
		bool spd_is_zero;
		double reward;
	};
	PenguinEnvSp CreateEnvironment(int frameSize);
	
}

#endif