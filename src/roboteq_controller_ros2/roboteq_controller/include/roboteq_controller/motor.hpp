#ifndef MOTOR_H
#define MOTOR_H

#include <rclcpp/rclcpp.hpp>
#include "roboteq_controller/serial_controller.hpp"
#include "configurator/gpio_sensor.hpp"
#include "configurator/motor_param.hpp"
#include "configurator/motor_pid.hpp"

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <diagnostic_updater/publisher.hpp>

#include "hardware_interface/system_interface.hpp"

//#include "roboteq_interfaces/msg/motor_status.hpp"
//#include "roboteq_interfaces/msg/control_status.hpp"
#include "joint_limits/joint_limits.hpp"
#include "urdf_parser/urdf_parser.h"
#include <filesystem>
#include "yaml-cpp/yaml.h"
#include <urdf/model.h>
#include <string>
#include <any>
#include <boost/lexical_cast.hpp>


namespace roboteq
{

typedef struct _motor_status {
    uint8_t amps_limit : 1;
    uint8_t motor_stalled : 1;
    uint8_t loop_error_detect : 1;
    uint8_t safety_stop_active : 1;
    uint8_t forward_limit_triggered : 1;
    uint8_t reverse_limit_triggered : 1;
    uint8_t amps_triggered_active : 1;
    uint8_t : 1;
} motor_status_t;

class Motor
{
public:
    /**
     * @brief Motor The motor definition and all ros controller initializations are collected in this place
     * @param nh The ROS private node handle
     * @param serial The serial controller
     * @param name The name of the motor
     * @param number The number in Roboteq board
     */
     Motor(serial_controller *serial, string name, unsigned int number, std::map<std::string, std::any>& roboteq_params);
    ~Motor();
    /**
     * @brief initializeMotor Initialization oh motor, this routine load parameter from ros server or load from roboteq board
     * @param load_from_board forse the load from roboteq board
     */
    void initializeMotor(bool load_from_board, std::map<std::string, std::any>& roboteq_params);
    /**
     * @brief setupLimits setup the maximum velocity, positio and effort
     * @param model the robot model
     */
    void setupLimits(urdf::Model model);
    /**
     * @brief writeCommandsToHardware Write a command to the hardware interface
     * @param period the period update
     */
    void writeCommandsToHardware(double command_);
    /**
     * @brief switchController Switch the controller from different type of ros controller
     * @param type the type of ros controller
     */
    //void switchController(string type);
    /**
     * @brief stopMotor Stop the motor
     */
    void stopMotor();
    /**
     * @brief getNumber The roboteq number
     * @return the number associated in the roboteq board
     */
    int getNumber() {
        return mNumber;
    }

    /**
     * @brief getName the name of the motor
     * @return the string with the name of the motor
     */
    string getName() {
        return mMotorName;
    }
    /**
     * @brief registerSensor register the sensor
     * @param sensor the sensor interface
     */
    void registerSensor(GPIOSensor* sensor, std::map<std::string, std::any>& roboteq_params);
    /**
     * @brief readVector Decode vector data list
     * @param fields field of measures
     */
    void readVector(std::vector<std::string> fields, double &mVel, double &mPos, double &mEffort);

    //hardware_interface::JointStateHandle joint_state_handle;
    //hardware_interface::JointHandle joint_handle;

    // Number of motor
    unsigned int mNumber;

    void setOperativeMode(std::map<std::string, std::any>& roboteq_params);  //shifted from motor_param.hpp to here
    
    void resetPosition(double position);
protected:
  /**
   * @param x Angular velocity in radians/s.
   * @return Angular velocity in RPM.
   */
  static double to_rpm(double x)
  {
    return x * 60 / (2 * M_PI);
  }

  /**
   * @param x Angular velocity in RPM.
   * @return Angular velocity in rad/s.
   */
  static double from_rpm(double x)
  {
    return x * (2 * M_PI) / 60;
  }

  /**
   * Conversion of radians to encoder ticks.
   *
   * @param x Angular position in radians.
   * @return Angular position in encoder ticks.
   */
  double to_encoder_ticks(double x);

  /**
   * Conversion of encoder ticks to radians.
   *
   * @param x Angular position in encoder ticks.
   * @return Angular position in radians.
   */
  double from_encoder_ticks(double x);
   
private:
    // Name of the motor
    string mMotorName;
    // Serial controller communication
    serial_controller *mSerial;
    // State of the motor
    double position_, max_position_;
    double velocity_, max_velocity_;
    double effort_, max_effort_;
    double command_;
    // Motor reduction and ratio
    double reduction_;
    double ratio_;
    // max RPM
    double max_rpm_ = 0;
    int32_t initial_encoder_pos = 0;
    bool initial_encoder_pos_set = false;

    int _control_mode;
    motor_status_t status_;

    rclcpp::Logger logger_ = rclcpp::get_logger("ros2_control_node");
    

    MotorParamConfigurator* parameter;
    MotorPIDConfigurator* pid_velocity;
    MotorPIDConfigurator* pid_torque;
    MotorPIDConfigurator* pid_position;

    GPIOSensor* _sensor;
};

}



#endif