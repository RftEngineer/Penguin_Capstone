#include "penguin_header.h"
#include "DQN.h"

namespace dqn {
	
	cv::Mat hwnd2mat(){
		HDC hwindowDC, hwindowCompatibleDC;

		int height, width, srcheight, srcwidth;
		cv::Mat src;
		BITMAPINFO dib_define;

		hwindowDC = GetDC(NULL);
		hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
		SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

		srcheight = g_maxY - g_minY;
		srcwidth = g_maxX - g_minX;
		height = g_maxY - g_minY;  //change this to whatever size you want to resize to
		width = g_maxX - g_minX;

		src.create(height, width, CV_8UC4);

		// create a bitmap

		HBITMAP h_bitmap = CreateCompatibleBitmap(hwindowDC, width, height);
		dib_define.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		dib_define.bmiHeader.biWidth = g_maxX - g_minX;
		dib_define.bmiHeader.biHeight = -(g_maxY - g_minY);
		dib_define.bmiHeader.biPlanes = 1;
		dib_define.bmiHeader.biBitCount = 32;
		dib_define.bmiHeader.biCompression = BI_RGB;
		dib_define.bmiHeader.biSizeImage = ((((g_maxX - g_minX) * 24 + 31) & ~31) >> 3) * (g_maxY - g_minY);
		dib_define.bmiHeader.biXPelsPerMeter = 0;
		dib_define.bmiHeader.biYPelsPerMeter = 0;
		dib_define.bmiHeader.biClrImportant = 0;
		dib_define.bmiHeader.biClrUsed = 0;
	
		// use the previously created device context with the bitmap
		SelectObject(hwindowCompatibleDC, h_bitmap);
		// copy from the window device context to the bitmap device context
		StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, g_minX, g_minY, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
		GetDIBits(hwindowCompatibleDC, h_bitmap, 0, height, src.data, (BITMAPINFO *)&dib_define, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow
		// avoid memory leak
		DeleteObject(h_bitmap); DeleteDC(hwindowCompatibleDC); ReleaseDC(NULL, hwindowDC);
		//DeleteObject(hbwindow); DeleteDC(hwindowCompatibleDC); ReleaseDC(hwnd, hwindowDC);
		//LOG(INFO) << "Capture Complete - need to transfer cv::mat";
		return src;
	}
		
	std::string PrintQValue(
		const std::vector<float>& q_values, const ActionVec& actions){
		assert(!q_values.empty());
		assert(!actions.empty());
		assert(q_values.size() == actions.size());
		std::ostringstream actions_buf;
		std::ostringstream q_values_buf;
		for (auto i = 0; i < q_values.size(); ++i)
		{
			const auto a_str = dqn::DQN::action_to_string(actions[i]);
			const auto q_str = std::to_string(q_values[i]);
			const auto column_size = (a_str.size()>q_str.size()) ? a_str.size()+1 : q_str.size()+1; //std::max(a_str.size(), q_str.size()) + 1;
			//max collision:Header defines different max
			//column_size += 1;
			actions_buf.width(column_size);
			actions_buf << a_str;
			q_values_buf.width(column_size);
			q_values_buf << q_str;
		}
		actions_buf << std::endl;
		q_values_buf << std::endl;
		return actions_buf.str() + q_values_buf.str();
	}

	ActionCode dqn::DQN::SelectAction(const State& frames, const double epsilon) 
	{
		return SelectActions(InputStateBatch{ { frames } }, epsilon)[0];
	}

	ActionVec dqn::DQN::SelectActions(
		const InputStateBatch& frames_batch,
		const double epsilon) {
		CHECK(epsilon <= 1.0 && epsilon >= 0.0);
		CHECK_LE(frames_batch.size(), miniBatchSize);
		ActionVec actions(frames_batch.size());
		if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine_) < epsilon) {
			// Select randomly
			for (int i = 0; i < actions.size(); ++i) {
				const auto random_idx = std::uniform_int_distribution<int>
					(0, legal_actions_.size() - 1)(random_engine_);
				actions[i] = legal_actions_[random_idx];
			}
		}
		else {
			// Select greedily
			std::vector<ActionValue> actions_and_values =
				SelectActionGreedily(target_net_, frames_batch);
			CHECK_EQ(actions_and_values.size(), actions.size());
			for (int i = 0; i<actions_and_values.size(); ++i) {
				actions[i] = actions_and_values[i].action;
			}
		}
		return actions;
	}
	
	const State Transition::GetNextState() const {

		//  Create the s(t+1) states from the experience(t)'s

		if (next_frame_ == nullptr) {
			// Terminal state so no next_observation, just return current state
			return state_;
		}
		else {
			State state_clone;

			for (int i = 0; i < inputFrameCount - 1; ++i)
				state_clone[i] = state_[i + 1];
			state_clone[inputFrameCount - 1] = next_frame_;
			return state_clone;
		}

	}
	template <typename Dtype>
	void HasBlobSize(caffe::Net<Dtype>& net,
		const std::string& blob_name,
		const std::vector<int> expected_shape) {
		net.has_blob(blob_name);
		const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
		const std::vector<int>& blob_shape = blob.shape();
		CHECK_EQ(blob_shape.size(), expected_shape.size());
		CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
			expected_shape.begin()));
	}
	ActionValue dqn::DQN::SelectActionGreedily(
		NetSp net,
		const State& last_frames) {
		//LOG(INFO) << "Return 1 first";
		return SelectActionGreedily(net, InputStateBatch{ { last_frames } }).front();
	}

	void dqn::DQN::AddTransition(const Transition& transition) {
		if (replay_memory_.size() == replay_memory_capacity_) {
			replay_memory_.pop_front();
		}
		replay_memory_.push_back(transition);
	}

	void dqn::DQN::LoadTrainModel(const std::string& model_bin) {
		net_->CopyTrainedLayersFrom(model_bin);
	}
	std::vector<ActionValue> dqn::DQN::SelectActionGreedily(
		NetSp net,
		const InputStateBatch& last_frames_batch) {
		//LOG(INFO) << "Hello" << last_frames_batch.size()<<"   "<<miniBatchSize;
		assert(last_frames_batch.size() <= miniBatchSize);
		
		std::array<float, miniBatchDataSize> frames_input;
		for (auto i = 0; i < last_frames_batch.size(); ++i) {
			// Input frames to the net and compute Q values for each legal actions
			for (auto j = 0; j < inputFrameCount; ++j) {
				const auto& frame_data = last_frames_batch[i][j];
				//LOG(INFO) << i << "  "<< j << "  " << frame_data;
				std::copy(
					frame_data->begin(),
					frame_data->end(),
					frames_input.begin() + (i * inputDataSize) +
					(j * croppedFrameDataSize));

			}
		}
		//LOG(INFO) << "SAG Step1  "<< frames_input.size();
		
		InputDataIntoLayers(net, frames_input, dummy_input_data_, dummy_input_data_);
		net->ForwardPrefilled(nullptr);

		std::vector<ActionValue> results;
		results.reserve(last_frames_batch.size());
		CHECK(net->has_blob(q_values_blob_name));
		const auto q_values_blob = net->blob_by_name(q_values_blob_name);
		for (auto i = 0; i < last_frames_batch.size(); ++i) {
			// Get the Q values from the net
			const auto action_evaluator = [&](ActionCode action) {
				const auto q = q_values_blob->data_at(i, static_cast<int>(action), 0, 0);
				assert(!std::isnan(q));
				return q;
			};
			std::vector<float> q_values(legal_actions_.size());
			std::transform(
				legal_actions_.begin(),
				legal_actions_.end(),
				q_values.begin(),
				action_evaluator);

			// Select the action with the maximum Q value
			const auto max_idx =
				std::distance(
				q_values.begin(),
				std::max_element(q_values.begin(), q_values.end()));
			results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
		}
		//LOG(INFO) << "SelectActionGreedily success..";
		return results;	
	}
	void dqn::DQN::Initialize() {

		std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);
		// Initialize net and solver
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
		solver_.reset(caffe::GetSolver<float>(solver_param));//New class 

		net_ = solver_->net();
		InitNet(net_);
		
		CloneTrainingNetToTargetNet();
				
		HasBlobSize(*net_, train_frames_blob_name, { miniBatchSize,
			inputFrameCount, croppedFrameSize, croppedFrameSize });
		HasBlobSize(*net_, target_blob_name, { miniBatchSize, outputCount, 1, 1 });
		HasBlobSize(*net_, filter_blob_name, { miniBatchSize, outputCount, 1, 1 });
		LOG(INFO) << "Finished " << net_->name() << " Initialization";
	}
		

	void dqn::DQN::InitNet(NetSp net) {
		const auto target_input_layer =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
			net->layer_by_name(target_layer_name));
		CHECK(target_input_layer);
		target_input_layer->Reset(const_cast<float*>(dummy_input_data_.data()),
			const_cast<float*>(dummy_input_data_.data()),
			target_input_layer->batch_size());
		const auto filter_input_layer =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
			net->layer_by_name(filter_layer_name));
		CHECK(filter_input_layer);
		filter_input_layer->Reset(const_cast<float*>(dummy_input_data_.data()),
			const_cast<float*>(dummy_input_data_.data()),
			filter_input_layer->batch_size());
	}

	void dqn::DQN::CloneNet(NetSp net) {
		caffe::NetParameter net_param;
		net->ToProto(&net_param);
		net_param.mutable_state()->set_phase(net->phase());
		if (target_net_ == nullptr) {
			target_net_.reset(new caffe::Net<float>(net_param));
		}
		else {
			target_net_->CopyTrainedLayersFrom(net_param);
		}
		InitNet(target_net_);
	}

	void dqn::DQN::Update(){
		if (verbose_)
			LOG(INFO) << "iteration: " << current_iteration() << std::endl;
		//LOG(INFO) << "Update Query!";
		if (current_iteration() >= last_clone_iter_ + clone_frequency_) {
			LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
			CloneTrainingNetToTargetNet();
			last_clone_iter_ = current_iteration();
		}

		std::vector<int> transitions;
		transitions.reserve(miniBatchSize);
		for (auto i = 0; i < miniBatchSize; ++i) {
			const auto random_transition_idx =
				std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
				random_engine_);
			transitions.push_back(random_transition_idx);
		}

		//LOG(INFO) << "Step1 Pass";

		// Compute target values: max_a Q(s',a)
		std::vector<State> target_last_frames_batch;
		for (const auto idx : transitions) {
			const auto& transition = replay_memory_[idx];
			if (transition.is_terminal()) {
				continue;
			}
			target_last_frames_batch.push_back(transition.GetNextState());
		}
		//LOG(INFO) << "Step2 pass";
		const auto actions_and_values =
			SelectActionGreedily(target_net_, target_last_frames_batch);
		////Warning. Please check Visual Studio Stack size.. It needs at least 4Mb or more.

		FramesLayerInputData frames_input;
		TargetLayerInputData target_input;
		FilterLayerInputData filter_input;
		std::fill(target_input.begin(), target_input.end(), 0.0f);
		std::fill(filter_input.begin(), filter_input.end(), 0.0f);
		auto target_value_idx = 0;
		//LOG(INFO) << "Step1 Pass";
		//
		for (auto i = 0; i < miniBatchSize; ++i) {
			const auto& transition = replay_memory_[transitions[i]];
			const auto action = transition.GetAction();
			const auto reward = transition.GetReward();
			assert(reward >= -1.0 && reward <= 1.0);
			const auto target = transition.is_terminal() ?
			reward :
				   reward + gamma_ * actions_and_values[target_value_idx++].q_value;
			assert(!std::isnan(target));
			target_input[i * outputCount + static_cast<int>(action)] = target;
			filter_input[i * outputCount + static_cast<int>(action)] = 1;
			if (verbose_)
				VLOG(1) << "filter:" << action_to_string(action)
				<< " target:" << target;
			for (auto j = 0; j < inputFrameCount; ++j) {
				const State& state = transition.GetState();
				const auto& frame_data = state[j];
				std::copy(
					frame_data->begin(),
					frame_data->end(),
					frames_input.begin() + i * inputDataSize +
					j * croppedFrameDataSize);
			}
		}
		InputDataIntoLayers(net_, frames_input, target_input, filter_input);
		solver_->Step(1);
		//std::cout << "123" << std::endl;
		//LOG(INFO) << "Step is operated.";
	}
	void dqn::DQN::InputDataIntoLayers(NetSp net,
		const FramesLayerInputData& frames_input,
		const TargetLayerInputData& target_input,
		const FilterLayerInputData& filter_input) {

		const auto frames_input_layer =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
			net->layer_by_name(frames_layer_name));
		CHECK(frames_input_layer);

		frames_input_layer->Reset(const_cast<float*>(frames_input.data()),
			const_cast<float*>(frames_input.data()),
			frames_input_layer->batch_size());

		if (net == net_) { // training net?
			const auto target_input_layer =
				boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net->layer_by_name(target_layer_name));
			CHECK(target_input_layer);
			target_input_layer->Reset(const_cast<float*>(target_input.data()),
				const_cast<float*>(target_input.data()),
				target_input_layer->batch_size());
			const auto filter_input_layer =
				boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net->layer_by_name(filter_layer_name));
			CHECK(filter_input_layer);
			filter_input_layer->Reset(const_cast<float*>(filter_input.data()),
				const_cast<float*>(filter_input.data()),
				filter_input_layer->batch_size());
		}

	}

	cv::Mat gameover = cv::imread("C:/capstone/pic/gameover.bmp");
	cv::Mat spd = cv::imread("C:/capstone/pic/spd.bmp");
	cv::Mat spd1 = cv::imread("C:/Capstone/pic/spd_new1.bmp");
	cv::Mat gameover1 = cv::imread("C:/Capstone/pic/gameover_new1.bmp");
	
	
}