//Caffe Train Mode

//#include <cuda_runtime.h>

#include "penguin_header.h"
#include "DQN.h"
#include "GameEnv.h"


DEFINE_bool(verbose, false, "verbose output");
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in evaluation mode");
DEFINE_int32(repeat_games, 1, "Number of games played in evaluation mode");
DEFINE_int32(memory, 5000000, "Capacity of replay memory");//Original is 500,000
DEFINE_int32(explore, 1000000, "Number of iterations needed for epsilon to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor used by Q value update.");
DEFINE_int32(memory_threshold, 100, "Enough amount of transitions to start "
	"learning");
DEFINE_int32(steps_per_epoch, 5000, "Number of training steps per epoch");
DEFINE_int32(screen_size, 1, "0 : 1x, 1: 3x");

DEFINE_string(solver_file, "C:/Capstone/proto/penguin_solver.prototxt", "solver_prototxt file");
DEFINE_string(net_file, "C:/Capstone/proto/penguin_net.prototxt", "net_prototxt file");
DEFINE_string(snapshot, "",	"Optional; the snapshot solver state to resume training.");
DEFINE_string(model, "", "Model file to load");

DEFINE_double(epsilon, 0.05, "Exploration used for evaluation.");
DEFINE_int32(learn_start, 50000, "Number of iteration before learn starts.");
DEFINE_int32(history_size, 1000000, "Number of transitions stored in replay memory.");
DEFINE_int32(update_freq, 4, "Number of actions taken between successive SGD updates.");
DEFINE_int32(frame_skip, 4, "Number of frames skipped between action selections.");
DEFINE_int32(clip_reward, 1, "Whether reward will be clipped to 1, 0, or -1 according to its sign.");
DEFINE_int32(game_status, 0, "Status 0 : Not on work, Status 1: Run, Status 2: Game Over ");
DEFINE_int32(program_mode, 0, "Mode 0 : Training mode, Mode 1: Run Mode");


double CalculateEpsilon(const int iter){
	if (iter < FLAGS_explore)
		return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
	else
		return 0.1;

}

double play(dqn::PenguinEnvSp envSp, dqn::DQN* dqn, const double epsilon, const bool update)
{
	assert(!envSp->EpisodeOver());
	time_t _t1;
	std::deque<dqn::FrameDataSp> past_frames;
	cv::Mat T1;
	envSp->keyReset();
	cv::Mat spdSrc;
	cv::Mat overSrc;
	cv::Mat clearSrc_1;
	if (FLAGS_screen_size == 0)
	{
		spdSrc = cv::imread("C:/Capstone/pic/spd.bmp");
		cv::cvtColor(spdSrc, spdSrc, CV_BGR2GRAY);
		overSrc = cv::imread("C:/Capstone/pic/gameover.bmp");
		cv::cvtColor(overSrc, overSrc, CV_BGR2GRAY);
		clearSrc_1 = cv::imread("C:/Capstone/pic/clear_1x.bmp");
		cv::cvtColor(clearSrc_1, clearSrc_1, CV_BGR2GRAY);
	}
	else if (FLAGS_screen_size == 1)
	{
		spdSrc = cv::imread("C:/Capstone/pic/spd_new1.bmp");
		cv::cvtColor(spdSrc, spdSrc, CV_BGR2GRAY);
		overSrc = cv::imread("C:/Capstone/pic/gameover_new1.bmp");
		cv::cvtColor(overSrc, overSrc, CV_BGR2GRAY);
		clearSrc_1 = cv::imread("C:/Capstone/pic/clear_3x_1.bmp");
		cv::cvtColor(clearSrc_1, clearSrc_1, CV_BGR2GRAY);
	}
	
	//LOG(INFO) << "Complete to make screenshot mat!";
	//double tmp_reward__ = 0.0;
	auto total_score = 0.0; // total score parts.
	auto cnt = time(&_t1);//
	//LOG(INFO) << "Process Lockon";
	for (auto frame = 0; !envSp->EpisodeOver(); ++frame) {
		auto cnt_time = time(&_t1);
		//std::cout << cnt << std::endl;
		if (FLAGS_verbose)
			LOG(INFO) << "frame: " << frame;
		T1 = dqn::hwnd2mat();//get image
		cv::Mat T5 = T1;
		cv::cvtColor(T5, T5, CV_BGR2GRAY);
		cv::imwrite("C:/Capstone/Test5.png", T5);
		const auto current_frame = envSp->PreprocessScreen(T1);
		cv::Mat T2, T3, T4;
		if (FLAGS_screen_size == 0)
		{
			T2 = envSp->setRoi(T5, 186, 14, 196, 22);//Speed becomes zero.
			T3 = envSp->setRoi(T5, 83, 61, 91, 69);//GameOver
			T4 = envSp->setRoi(T5, 117, 153, 130, 160);//Clear
		}
		else if (FLAGS_screen_size == 1)
		{
			T2 = envSp->setRoi(T5, 530, 13, 580, 38);//Speed becomes zero.
			T3 = envSp->setRoi(T5, 220, 156, 241, 179);//GameOver
			T4 = envSp->setRoi(T5, 321, 431, 357, 450);//Clear
		}
		//if (envSp->checkImg(T2, spdSrc, T5) || envSp->checkImg(T3, overSrc, T5))
		if (envSp->checkImg(T3,overSrc,T5))
		{
			envSp->setEpisodeOver(true);
			total_score -= 500;
			break;
		}
		else if (envSp->checkImg(T2, spdSrc, T5))
		{
			total_score -= 20;
			cnt = cnt_time;
		}
		else if (envSp->checkImg(T4, clearSrc_1, T5))
		{
			total_score += 1000;
			envSp->setEpisodeOver(true);
			break;
		}
	
		past_frames.push_back(current_frame);
		
		//LOG(INFO) << "Working!";
		//LOG(INFO) << past_frames.size() << " " << dqn::inputFrameCount;
		if (past_frames.size() < dqn::inputFrameCount){
			
			envSp->ActNoop(cnt_time-cnt);
		}
		else {
			if (past_frames.size() > dqn::inputFrameCount){
				past_frames.pop_front();
			}
			dqn::State input_frames;
			std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
			const auto action = dqn->SelectAction(input_frames, epsilon);
			
			auto immediate_score = envSp->Act(action,cnt_time-cnt);
			total_score += immediate_score;

			const auto reward =
				immediate_score == 0 ?
				0 :
				immediate_score /= std::abs(immediate_score);
			if (update) {
				// Add the current transition to replay memory
				//std::cout << "Step" << std::endl;
				cv::Mat T2;
				T2 = dqn::hwnd2mat();
				//LOG(INFO) << "Reward: " << reward;
				const auto transition = envSp->EpisodeOver() ?//dqn::EpisodeOver() ?
					dqn::Transition(input_frames, action, reward, nullptr) :
					dqn::Transition(
					input_frames,
					action,
					reward,
					envSp->PreprocessScreen(T2));
				dqn->AddTransition(transition);
				//LOG(INFO) << cnt;
				// If the size of replay memory is enough, update DQN
				if (dqn->memory_size() > FLAGS_memory_threshold) {
					//std::cout << "Update" << std::endl;
					dqn->Update();
				}
			}
		}
	}
	//dqn::Reset();
	envSp->Reset();
	return total_score;
}


int main(int argc, char** argv)
{
	
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	caffe::GlobalInit(&argc, &argv);
	google::LogToStderr();
	
	
	caffe::Caffe::SetDevice(0);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);
	
	dqn::PenguinEnvSp penguinEnvP = dqn::CreateEnvironment(FLAGS_screen_size);

	const dqn::PenguinEnv::ActionVec legal_actions =
		penguinEnvP->GetMinimalActionSet();
	//dqn::ActionVec legal_actions;
	//for (auto i = 0; i < 5; i++)
	//{
	//	legal_actions.push_back(static_cast<dqn::ActionCode>(i));
	//}
	HWND hWnd = GetConsoleWindow();
	SetWindowPos(hWnd, HWND_TOP, 800,200,900,300, SWP_NOSIZE);
	
	dqn::DQN penguin_net(legal_actions, FLAGS_solver_file, FLAGS_memory, FLAGS_gamma, FLAGS_verbose);
	penguin_net.Initialize();
	
	//penguin_net.Update();
	//const auto epsilon = CalculateEpsilon(penguin_net.current_iteration());
	//auto train_score = play(penguinEnvP,&penguin_net, epsilon, true);
	
	//dqn::DQN *penguin_net1(legal_actions, FLAGS_solver_file, FLAGS_memory, FLAGS_gamma, FLAGS_verbose);
	//penguin_net1->Initialize();

	//LOG(INFO) << "Evaluate: " << FLAGS_evaluate;
	//FLAGS_evaluate = true;
	//FLAGS_model = "C:/Capstone/dqn_iter_630000.caffemodel";
	//FLAGS_snapshot = "C:/Capstone/dqn_iter_630000.solverstate";
	//FLAGS_repeat_games = 4;
	if (!FLAGS_model.empty()){
		LOG(INFO) << "Loading " << FLAGS_model;
	}

	if (FLAGS_evaluate){
		LOG(INFO) << "Run on Evaluate Mode";
		penguin_net.LoadTrainModel(FLAGS_model);
		auto total_score = 0.0;
		for (auto i = 0; i < FLAGS_repeat_games; ++i)
		{
			LOG(INFO) << "game : ";
			//const auto score = play(penguin_net, FLAGS_evaluate_with_epsilon, false);
			//const auto score = play(penguinEnvP,&penguin_net, FLAGS_evaluate_with_epsilon, false);
			const auto epsilon = CalculateEpsilon(penguin_net.current_iteration());
			auto score = play(penguinEnvP, &penguin_net, epsilon, false);
			LOG(INFO) << "score: " << score;
			total_score += score;
		}
		LOG(INFO) << "total_score: " << total_score;
		return 0;
	}

	double total_score = 0.0;
	double epoch_total_score = 0.0;
	int epoch_episode_count = 0.0;
	double total_time = 0.0;
	int next_epoch_boundry = FLAGS_steps_per_epoch;
	double running_average = 0.0;
	double plot_average_discount = 0.05;
	
	std::ofstream training_data("C:/Capstone/penguin_log.csv");
	training_data << "Penguin_net," << FLAGS_steps_per_epoch
		<< ",,," << std::endl;
	training_data << "Epoch,Epoch avg score,Hours training,Number of episodes"
		",episodes in epoch" << std::endl;
	//LOG(INFO) << "Step1 on main";
	//const auto epsilon = CalculateEpsilon(penguin_net.current_iteration());
	//LOG(INFO) << "Step2 on main";
	//
	for (auto episode = 0;; episode++) {
		caffe::Timer run_timer;
		run_timer.Start();
		LOG(INFO) << "Timer running";
		epoch_episode_count++;
		const auto epsilon = CalculateEpsilon(penguin_net.current_iteration());
		auto train_score = play(penguinEnvP,&penguin_net, epsilon, true);
		//auto train_score = std::log(episode + 1);

		epoch_total_score += train_score;
		if (penguin_net.current_iteration() > 0)
			total_time += run_timer.MilliSeconds();
		LOG(INFO) << "training score(" << episode << "): "
			<< train_score << std::endl;

		if (episode == 0)
			running_average = train_score;
		else
			running_average = train_score*plot_average_discount + running_average*(1.0 - plot_average_discount);
		
		LOG(INFO) << "Current Iteration: " << penguin_net.current_iteration();
		
		if (penguin_net.current_iteration() >= next_epoch_boundry){
			double hours = total_time / 1000. / 3600.;
			int epoc_number = static_cast<int>(
				(next_epoch_boundry) / FLAGS_steps_per_epoch);
			LOG(INFO) << "epoch(" << epoc_number
				<< ":" << penguin_net.current_iteration() << "): "
				<< "average score " << running_average << " in "
				<< hours << " hour(s)";


			if (penguin_net.current_iteration()){
				auto hours_for_million = hours / (penguin_net.current_iteration() / 1000000.0);
				LOG(INFO) << "Estimated Time for 1 million iterations: "
					<< hours_for_million
					<< " hours";
			}
			training_data << epoc_number << ", " << running_average << ", " << hours
				<< ", " << episode << ", " << epoch_episode_count << std::endl;

			epoch_total_score = 0.0;
			epoch_episode_count = 0;

			while (next_epoch_boundry < penguin_net.current_iteration())
				next_epoch_boundry += FLAGS_steps_per_epoch;
		}
		
	}
	
	training_data.close();
	
}
