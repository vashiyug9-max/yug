#include "roboteq_controller/motor.hpp"
#include <boost/algorithm/algorithm.hpp>
#include <boost/algorithm/string.hpp>


namespace roboteq {

    Motor::Motor(serial_controller *serial, string name, unsigned int number, std::map<std::string, std::any>& roboteq_params): mSerial(serial)
    {
        // Initialize all variables
        mNumber = number;
        mMotorName = name;
        command_ = 0;
        // Reset variables
        position_ = 0;
        velocity_ = 0;
        effort_ = 0;
        // Initialize control mode
        _control_mode = -1;
        // Initialize reduction and get rati
        reduction_ = 0;
        //replaced
        //ratio_ = std::any_cast<int>(roboteq_params[name+ "_" +"ratio"]);
        //added
        RCLCPP_INFO(logger_, "Motor %i, ratio: %f",  mNumber ,ratio_);

        max_rpm_ = std::any_cast<double>(roboteq_params[name+ "_" +"maxSpeed"]);
        RCLCPP_INFO(logger_, "Motor %i, rpm: %f", mNumber,  max_rpm_);

        // Dynamic reconfigure connot be used in ROS2 hardware interface
        parameter = new MotorParamConfigurator(serial, mMotorName, number);
        pid_velocity = new MotorPIDConfigurator(serial, mMotorName, "velocity", number);
        pid_torque = new MotorPIDConfigurator(serial, mMotorName, "torque", number);
        pid_position = new MotorPIDConfigurator(serial, mMotorName, "position", number);
    }

    Motor::~Motor() {
        delete parameter;
        delete pid_velocity;
        delete pid_torque;
        delete pid_position;
    }



    void Motor::setupLimits(urdf::Model model)
    {
        joint_limits::JointLimits limits;
        joint_limits::SoftJointLimits soft_limits;

        bool state = true;

        // Manual value setting
        limits.has_velocity_limits = true;
        limits.max_velocity = 5.0;
        limits.has_effort_limits = true;
        limits.max_effort = 2.0;


    }

    void Motor::registerSensor(GPIOSensor* sensor, std::map<std::string, std::any>& roboteq_params)
    {
        _sensor = sensor;
        //replaced
        //reduction_ = std::any_cast<int>(roboteq_params[mMotorName + "_ratio"]);
        //added
        reduction_ =
            std::any_cast<double>(roboteq_params.at(mMotorName + "_ratio"));
        //until here
        RCLCPP_INFO(logger_, "reduction_ while registering sensor from yaml: %f ", reduction_);
        reduction_ = _sensor->getConversion(reduction_, roboteq_params);
        RCLCPP_INFO(logger_, "reduction_ after converison: %f ", reduction_);
    }


    void Motor::initializeMotor(bool load_from_board, std::map<std::string, std::any>& roboteq_params)
    {
        parameter->initConfigurator(load_from_board, roboteq_params);
        max_rpm_ = std::any_cast<double>(roboteq_params[mMotorName+"_maxSpeed"]);
        RCLCPP_INFO(logger_, "max_rpm_ : %f", max_rpm_);
        _control_mode = parameter->getOperativeMode();
        RCLCPP_INFO(logger_, "_control_mode : %d", _control_mode);
        
        bool tmp_pos = load_from_board & ((_control_mode == 2) || (_control_mode == 3) || (_control_mode == 4));
        bool tmp_vel = load_from_board & ((_control_mode == 1) || (_control_mode == 6));
        bool tmp_tor = load_from_board & (_control_mode == 5);

        pid_position->initConfigurator(tmp_pos, roboteq_params);
        pid_velocity->initConfigurator(tmp_vel, roboteq_params);
        pid_torque->initConfigurator(tmp_tor, roboteq_params);

        // stop the motor
        stopMotor();
    }

    void Motor::setOperativeMode(std::map<std::string, std::any>& roboteq_params)
    {
        // Update operative mode
        int pid_vel = std::any_cast<int>(roboteq_params[mMotorName+"_pid_closedLoopVelocity"]);
        RCLCPP_INFO(logger_, "setting operative mode to %d", pid_vel);
        parameter->setOperativeMode(pid_vel);
        _control_mode = pid_vel;
    }

    void Motor::resetPosition(double position)
    {
        // Send reset position
        double enc_conv = to_encoder_ticks(position);
        mSerial->command("C ", std::to_string(mNumber) + " " + std::to_string(enc_conv));
    }

    void Motor::stopMotor()
    {
        // set to zero the reference
        mSerial->command("G ", std::to_string(mNumber) + " 0");
        // Stop motor [pag 222]
        mSerial->command("MS", std::to_string(mNumber));
    }

    /**
     * Conversion of radians to encoder ticks.
     *
     * @param x Angular position in radians.
     * @return Angular position in encoder ticks.
     */
    double Motor::to_encoder_ticks(double x)
    {
        // Return the value converted
        return x * (reduction_) / (2 * M_PI);
    }

    /**
     * Conversion of encoder ticks to radians.
     *
     * @param x Angular position in encoder ticks.
     * @return Angular position in radians.
     */
    double Motor::from_encoder_ticks(double x)
    {
        // Return the value converted
        RCLCPP_INFO(logger_, "x: %f, reduction_, : %f", x, reduction_ );
        return x * (2 * M_PI) / (reduction_);
    }
    //replaced
    //void Motor::readVector(std::vector<std::string> fields, double &mVel, double &mPos, double &mEffort) {
    //    double position, vel, volts, amps_motor;
    //added
    void Motor::readVector(
        std::vector<std::string> fields,
        double &mVel,
        double &mPos,
        double &mEffort)
    {
        // Preserve the last valid state if this feedback cannot be decoded.
        mVel = velocity_;
        mPos = position_;
        mEffort = effort_;

        if (fields.size() != 10)
        {
            RCLCPP_WARN(
                logger_,
                "Motor %u (%s): expected 10 feedback fields, received %zu. "
                "Keeping previous state.",
                mNumber,
                mMotorName.c_str(),
                fields.size()
            );
            return;
        }

        double position, vel, volts, amps_motor;
        //until here
        // Scale factors as outlined in the relevant portions of the user manual, please
        // see mbs/script.mbs for URL and specific page references.
        try
        {
            // reference command FM <-> _MOTFLAG [pag. 246]
            unsigned char status = boost::lexical_cast<unsigned int>(fields[0]);
            RCLCPP_INFO(logger_, "Motor %d status: %u", mNumber, status);
            memcpy(&status_, &status, sizeof(status));
            // reference command M <-> _MOTCMD [pag. 250]
            double cmd = boost::lexical_cast<double>(fields[1]) * max_rpm_ / 1000.0;

            // reference command F <-> _FEEDBK [pag. 244]
            vel = boost::lexical_cast<double>(fields[2]) * max_rpm_ / 1000.0;
            // reference command E <-> _LPERR [pag. 243]
            double loop_error = boost::lexical_cast<double>(fields[3]) * max_rpm_ / 1000.0;
            // reference command P <-> _MOTPWR [pag. 255]
            double pwm = boost::lexical_cast<double>(fields[4]);
            // reference voltage V <-> _VOLTS [pag. ---]
            volts = boost::lexical_cast<double>(fields[5]) / 10;
            // reference command A <-> _MOTAMPS [pag. 230]
            amps_motor = boost::lexical_cast<double>(fields[6]) / 10;
            // reference command BA <-> _BATAMPS [pag. 233]
            double amps_batt = boost::lexical_cast<double>(fields[7]) / 10;
            // Reference command CR <-> _RELCNTR [pag. 241]
            // To check and substitute with C
            // Reference command C <-> _ABCNTR [pag. ---]
            position = boost::lexical_cast<double>(fields[8]);
            if(!initial_encoder_pos_set){
                initial_encoder_pos = position;
                initial_encoder_pos_set = true;
            }
            // reference command TR <-> _TR [pag. 260]
            double track = boost::lexical_cast<long>(fields[9]);

            RCLCPP_INFO(logger_,"Motor %d: cmd=%.2f rpm, vel=%.2f rpm, loop_error=%.2f rpm, pwm=%.2f, volts=%.1f V, amps_motor=%.1f A, amps_batt=%.1f A, position=%.0f, track=%.0f",
                    mNumber, cmd, vel, loop_error, pwm, volts, amps_motor, amps_batt, position, track);

        }
        //replaced
        /*
        catch (std::bad_cast& e)
        {   //replaced
            //RCLCPP_WARN(logger_, " [%d] %s : Failure parsing feedback data. Dropping message. %s", mNumber, mMotorName, e.what());
            //added
            RCLCPP_WARN(
                logger_,
                " [%d] %s : Failure parsing feedback data. Dropping message. %s",
                mNumber,
                mMotorName.c_str(),
                e.what()
            );
            //until here
            // Load same values
            position = to_encoder_ticks(position_);
            vel = ratio_ * velocity_;
            volts = 0;
            amps_motor = 0;
        }
        */
        catch (const std::bad_cast& e)
        {
            RCLCPP_WARN(
                logger_,
                "Motor %u (%s): failure parsing feedback data: %s. "
                "Keeping previous state.",
                mNumber,
                mMotorName.c_str(),
                e.what()
            );
            position = to_encoder_ticks(position_);
            vel = ratio_ * velocity_;
            volts = 0;
            amps_motor = 0;
            return;
        }
        // Update position
        RCLCPP_INFO(logger_, "POSITION of encoder: %f", position-initial_encoder_pos);
        position_ = from_encoder_ticks(position-initial_encoder_pos);
        RCLCPP_INFO(logger_, "angular POSITION: %f", position_);
        mPos = position_;
        // Update velocity motor
        velocity_ = (vel / ratio_);
        mVel = velocity_;
        // Evaluate effort
        if(velocity_ != 0)
        {
            effort_ = ((volts * amps_motor) / velocity_) * ratio_;
        }
        else
        {
            effort_ = 0;
        }
        mEffort = effort_;
    }
    
    
    
    void Motor::writeCommandsToHardware(double command_)
    {   
        long long int roboteq_velocity = static_cast<long long int>(to_rpm(command_) / max_rpm_ * 1000.0);   //vel = command_ * 23.870146
        
        //RCLCPP_INFO(logger_, "Motor num: %d,  Velocity: %ld, val: %f", mNumber, roboteq_velocity, command_ );

        mSerial->command("G ", std::to_string(mNumber) + " " + std::to_string(roboteq_velocity));
        //replaced
        //RCLCPP_INFO(logger_, "G: %d,  Velocity: %ld", mNumber, roboteq_velocity);
        //added
        RCLCPP_INFO(
            logger_,
            "G: %d,  Velocity: %lld",
            mNumber,
            roboteq_velocity
        );
        //until here

    }

}