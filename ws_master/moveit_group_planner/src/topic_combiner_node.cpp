
#include "utilities.h"
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/header.hpp>
#include <rclcpp/qos.hpp>



class TopicCombinerNode : public rclcpp::Node
{
public:
  TopicCombinerNode()
  : Node("topic_combiner_node")
  {
    // Create subscribers for the input topics
    sub_1_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "ctrl_groups/b1/joint_states", rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT), std::bind(&TopicCombinerNode::topic1_callback, this, std::placeholders::_1));
    sub_2_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "ctrl_groups/r1/joint_states", rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT), std::bind(&TopicCombinerNode::topic2_callback, this, std::placeholders::_1));
    sub_3_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "ctrl_groups/r2/joint_states", rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT), std::bind(&TopicCombinerNode::topic3_callback, this, std::placeholders::_1));
    sub_4_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "ctrl_groups/s1/joint_states", rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT), std::bind(&TopicCombinerNode::topic4_callback, this, std::placeholders::_1));

    // Create publisher for the output topic
    pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
  }

private:
  void topic1_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    process_message(msg, 1);
  }

  void topic2_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    process_message(msg, 2);
  }

  void topic3_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    process_message(msg, 3);
  }

  void topic4_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    process_message(msg, 4);
  }

  void process_message(const sensor_msgs::msg::JointState::SharedPtr msg, int topic_number)
  {
    switch(topic_number)
    {
      case 1:
        input_1_ = msg;
        break;
      case 2:
        input_2_ = msg;
        break;
      case 3:
        input_3_ = msg;
        break;
      case 4:
        input_4_ = msg;
        break;
      default:
        break;
    }
    
    publish_combined_message();
  }

  void publish_combined_message()
  {
    if(input_1_ && input_2_ && input_3_ && input_4_)
    {
      sensor_msgs::msg::JointState combined_msg;
      
      // Use the header from one of the messages (e.g., input_1_)
      combined_msg.header = input_1_->header;
      
      // Concatenate the data from all inputs
      combined_msg.name.insert(combined_msg.name.end(), input_1_->name.begin(), input_1_->name.end());
      combined_msg.name.insert(combined_msg.name.end(), input_2_->name.begin(), input_2_->name.end());
      combined_msg.name.insert(combined_msg.name.end(), input_3_->name.begin(), input_3_->name.end());
      combined_msg.name.insert(combined_msg.name.end(), input_4_->name.begin(), input_4_->name.end());

      combined_msg.position.insert(combined_msg.position.end(), input_1_->position.begin(), input_1_->position.end());
      combined_msg.position.insert(combined_msg.position.end(), input_2_->position.begin(), input_2_->position.end());
      combined_msg.position.insert(combined_msg.position.end(), input_3_->position.begin(), input_3_->position.end());
      combined_msg.position.insert(combined_msg.position.end(), input_4_->position.begin(), input_4_->position.end());

      combined_msg.velocity.insert(combined_msg.velocity.end(), input_1_->velocity.begin(), input_1_->velocity.end());
      combined_msg.velocity.insert(combined_msg.velocity.end(), input_2_->velocity.begin(), input_2_->velocity.end());
      combined_msg.velocity.insert(combined_msg.velocity.end(), input_3_->velocity.begin(), input_3_->velocity.end());
      combined_msg.velocity.insert(combined_msg.velocity.end(), input_4_->velocity.begin(), input_4_->velocity.end());

      combined_msg.effort.insert(combined_msg.effort.end(), input_1_->effort.begin(), input_1_->effort.end());
      combined_msg.effort.insert(combined_msg.effort.end(), input_2_->effort.begin(), input_2_->effort.end());
      combined_msg.effort.insert(combined_msg.effort.end(), input_3_->effort.begin(), input_3_->effort.end());
      combined_msg.effort.insert(combined_msg.effort.end(), input_4_->effort.begin(), input_4_->effort.end());

      pub_->publish(combined_msg);

      // Clear inputs after publishing
      input_1_.reset();
      input_2_.reset();
      input_3_.reset();
      input_4_.reset();
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_1_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_2_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_3_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_4_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;

  sensor_msgs::msg::JointState::SharedPtr input_1_;
  sensor_msgs::msg::JointState::SharedPtr input_2_;
  sensor_msgs::msg::JointState::SharedPtr input_3_;
  sensor_msgs::msg::JointState::SharedPtr input_4_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TopicCombinerNode>());
  rclcpp::shutdown();
  return 0;
}
