/**
 * Copyright (C) 2017, Raffaello Bonghi <raffaello@rnext.it>
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "configurator/gpio_analog.hpp"

#define PARAM_ANALOG_STRING "_analog"

GPIOAnalogConfigurator::GPIOAnalogConfigurator(roboteq::serial_controller *serial, std::vector<roboteq::Motor *> motor, string name, unsigned int number)
    : mSerial(serial)
    ,_motor(motor)
{
    // Find path param
    mName = name + PARAM_ANALOG_STRING + "_" + std::to_string(number);
    // Roboteq motor channel number
    mNumber = number;
    // Set false on first run
    setup_param = false;
}

void GPIOAnalogConfigurator::initConfigurator(bool load_from_board, std::map<std::string, std::any>& roboteq_params)
{
    // Check if is required load paramers
    if(load_from_board)
    {
        // Load parameters from roboteq
        getParamFromRoboteq(roboteq_params);
    }
}

void GPIOAnalogConfigurator::getParamFromRoboteq(std::map<std::string, std::any>& roboteq_params)
{
    try
    {
        // conversion AMOD
        string str_conversion = mSerial->getParam("AMOD", std::to_string(mNumber));
        int conversion = boost::lexical_cast<int>(str_conversion);
        // Set params
        roboteq_params[mName + "_conversion"] = conversion;

        // input AINA
        string str_pina = mSerial->getParam("AINA", std::to_string(mNumber));
        // Get AINA from roboteq board
        int emod = boost::lexical_cast<unsigned int>(str_pina);
        // 3 modes:
        // 0 - Unsed
        // 1 - Command
        // 2 - Feedback
        int command = (emod & 0b11);
        int motors = (emod - command) >> 4;
        int tmp1 = ((motors & 0b1) > 0);
        int tmp2 = ((motors & 0b10) > 0);
        if(tmp1)
        {
            for (vector<roboteq::Motor*>::iterator it = _motor.begin() ; it != _motor.end(); ++it)
            {
                roboteq::Motor* motor = ((roboteq::Motor*)(*it));
                if(motor->getNumber() == 1)
                {
                    motor->registerSensor(this, roboteq_params);
                    RCLCPP_INFO(logger_, "Register pulse input [%d] to: %s", mNumber, motor->getName());
                    break;
                }
            }
        }
        if(tmp2)
        {
            for (vector<roboteq::Motor*>::iterator it = _motor.begin() ; it != _motor.end(); ++it)
            {
                roboteq::Motor* motor = ((roboteq::Motor*)(*it));
                if(motor->getNumber() == 2)
                {
                    motor->registerSensor(this, roboteq_params);
                    RCLCPP_INFO(logger_, "Register pulse input [%d] to: %s", mNumber, motor->getName());
                    break;
                }
            }
        }

        // Set parameter
        roboteq_params[mName + "_inputUse"] = command;
        roboteq_params[mName + "_inputMotorOne"] = tmp1;
        roboteq_params[mName + "_inputMotorTwo"] = tmp2;

        // polarity APOL
        string str_polarity = mSerial->getParam("APOL", std::to_string(mNumber));
        int polarity = boost::lexical_cast<int>(str_polarity);
        // Set params
        roboteq_params[mName + "_conversionPolarity"] = polarity;

        // Input deadband ADB
        string str_deadband = mSerial->getParam("ADB", std::to_string(mNumber));
        int deadband = boost::lexical_cast<int>(str_deadband);
        // Set params
        roboteq_params[mName + "_inputDeadband"] = deadband;

        // Input AMIN
        string str_min = mSerial->getParam("AMIN", std::to_string(mNumber));
        double min = boost::lexical_cast<double>(str_min) / 1000;
        // Set params
        roboteq_params[mName + "_rangeInputMin"] = min;

        // Input AMAX
        string str_max = mSerial->getParam("AMAX", std::to_string(mNumber));
        double max = boost::lexical_cast<double>(str_max) / 1000;
        // Set params
        roboteq_params[mName + "_rangeInputMax"] = max;

        // Input ACTR
        string str_ctr = mSerial->getParam("ACTR", std::to_string(mNumber));
        double ctr = boost::lexical_cast<double>(str_ctr) / 1000;
        // Set params
        roboteq_params[mName + "_rangeInputCenter"] = ctr;

    } catch (std::bad_cast& e)
    {
        RCLCPP_WARN(logger_, "Failure parsing feedback data. Dropping message.%s", e.what());
    }
}

