#ifndef DQN_HPP_
#define DQN_HPP_
/*
Action List : 0: Nothing, 1: Up, 2: Down, 3: Left, 4: Right, 5: Jump

*/
#include "penguin_header.h"
#include "GameEnv.h"
namespace dqn {
	/*const auto g_minX = 6;
	const auto g_minY = 83;
	const auto g_maxX = 255;
	const auto g_maxY = 295;*/
	const auto g_minX = 46;
	const auto g_minY = 114;
	const auto g_maxX = 732;
	const auto g_maxY = 718;

	const auto rawFrameHeight = 212;
	const auto rawFrameWidth = 249;
	const auto croppedFrameSize = 84;
	const auto croppedFrameDataSize = 7056; // 84 * 84
	const auto inputFrameCount = 15;
	const auto inputDataSize = croppedFrameDataSize * inputFrameCount;
	const auto miniBatchSize = 32;
	const auto miniBatchDataSize = inputDataSize * miniBatchSize;
	const auto gamma = 0.95f;
	const auto outputCount = 18;

	const auto frames_layer_name = "frames_input_layer";
	const auto target_layer_name = "target_input_layer";
	const auto filter_layer_name = "filter_input_layer";

	const auto train_frames_blob_name = "frames";
	const auto test_frames_blob_name = "all_frames";
	const auto filter_blob_name = "filter";
	const auto target_blob_name = "target";
	const auto q_values_blob_name = "q_values";

	using FrameData = std::array<uint8_t, croppedFrameDataSize>;
	using FrameDataSp = std::shared_ptr<FrameData>;
	using State = std::array<FrameDataSp, inputFrameCount>;
	using InputStateBatch = std::vector<State>;

	using FramesLayerInputData = std::array<float, miniBatchDataSize>;
	using TargetLayerInputData = std::array<float, miniBatchSize*outputCount>;
	using FilterLayerInputData = std::array<float, miniBatchSize*outputCount>;

	typedef int ActionCode;
	typedef std::vector<int> ActionVec;
	typedef struct ActionValue {
		ActionValue(const ActionCode _action, const float _q_value) :
		action(_action), q_value(_q_value) {
		}
		const ActionCode action;
		const float q_value;
	} ActionValue;

	cv::Mat hwnd2mat();
	
	class Transition{
	public:
		Transition(const State state, ActionCode action,
			double reward, FrameDataSp next_frame) :
			state_(state),
			action_(action),
			reward_(reward),
			next_frame_(next_frame) {
		}

		bool is_terminal() const { return next_frame_ == nullptr; }

		const State GetNextState() const;

		const State& GetState() const { return state_; }

		ActionCode GetAction() const { return action_; }

		double GetReward() const { return reward_; }

	private:
		const State state_;
		ActionCode action_;
		double reward_;
		FrameDataSp next_frame_;
	};
	typedef std::shared_ptr<Transition> TransitionSp;
	
	class DQN {
	public:
		DQN(
			const ActionVec& legal_actions,
			const std::string& solver_param,
			const int replay_memory_capacity,
			const double gamma,
			const bool verbose) :
			legal_actions_(legal_actions),
			solver_param_(solver_param),
			replay_memory_capacity_(replay_memory_capacity),
			gamma_(gamma),
			verbose_(verbose),
			random_engine_(0),
			clone_frequency_(10000),
			last_clone_iter_(0) {
				
			}

		static std::string action_to_string(ActionCode a){ 
			return m_action_to_string(static_cast<Action>(a));
		}
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
		void Initialize();

		void LoadTrainModel(const std::string& model_file);

		ActionCode SelectAction(const State& frames, const double epsilon);
		
		void AddTransition(const Transition& transition);

		void Update();

		void CloneTrainingNetToTargetNet() { CloneNet(net_); }

		int memory_size() const { return replay_memory_.size(); }
		int current_iteration() const { return solver_->iter(); }
	private:
		using SolverSp = std::shared_ptr<caffe::Solver<float>>;
		using NetSp = boost::shared_ptr<caffe::Net<float>>;
		using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
		using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;

		ActionVec SelectActions(const InputStateBatch& frames_batch,
			const double epsilon);
		ActionValue SelectActionGreedily(NetSp net,
			const State& last_frames);
		std::vector<ActionValue> SelectActionGreedily(NetSp,
			const InputStateBatch& last_frames);


		const ActionVec legal_actions_;
		const int replay_memory_capacity_;
		const double gamma_;
		

		// clone the given net and store the result in clone_net_;
		void CloneNet(NetSp net);

		//Init target, filter layers;
		void InitNet(NetSp net);

		void InputDataIntoLayers(NetSp net,
			const FramesLayerInputData& frames_data,
			const TargetLayerInputData& target_data,
			const FilterLayerInputData& filter_data);

		std::deque<Transition> replay_memory_;
		NetSp net_;
		NetSp target_net_;
		SolverSp solver_;
		const std::string solver_param_;
		BlobSp q_value_blob_;
		const int clone_frequency_;
		int last_clone_iter_;

		MemoryDataLayerSp frames_input_layer_;
		MemoryDataLayerSp target_input_layer_;
		MemoryDataLayerSp filter_input_layer_;
		TargetLayerInputData dummy_input_data_;

		std::mt19937 random_engine_;
		bool verbose_;
	};

}

#endif