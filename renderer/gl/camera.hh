// Copyright 2017-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//File: camera.hh

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lib/timer.hh"
#include "lib/geometry.hh"
namespace {
const glm::vec3 WORLD_UP{0.f, 1.f, 0.f};
// TODO change DEFAULT_NEAR to 0.01
const float DEFAULT_NEAR = 0.1f,
            DEFAULT_FAR = 100.f;
const float DEFAULT_VERTICAL_FOV = 60.f;
} // namespace
namespace py=pybind11;

namespace render {

class Camera {
  public:
    glm::vec3 pos;
    glm::vec3 front;
    glm::vec3 up; // constant?
    glm::vec3 right;
    float yaw, pitch;
    float near, far;
    float vertical_fov;

    enum Movement : char {
        FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN
    };


    Camera(glm::vec3 pos, float yaw=-90.f, float pitch=0.f):
      pos{pos}, up{WORLD_UP}, yaw{yaw}, pitch{pitch}, near{DEFAULT_NEAR},
      far{DEFAULT_FAR}, vertical_fov{DEFAULT_VERTICAL_FOV} {
        updateDirection();
      }

    glm::mat4 getView() const
    {return glm::lookAt(pos, pos + front, up); }

    void shift(Movement dir, float dist);

    // difference in yaw and pitch
    void turn(float dyaw, float dpitch);

    // If you modify yaw or pitch manually, you need to call this function after the modification.
    void updateDirection() {
      computeFront();
      right = glm::normalize(glm::cross(front, up));
    }

    glm::mat4 getCameraMatrix(const Geometry& geo) const {
      glm::mat4 projection = glm::perspective(
          glm::radians(vertical_fov),
          (float)geo.w / geo.h, near, far);
      glm::mat4 view = getView();
      return projection * view;
    }

    py::array_t<float> getCameraMatrixNumpy(const Geometry& geo) const {
      glm::mat4 projection = glm::perspective(
          glm::radians(vertical_fov),
          (float)geo.w / geo.h, near, far);
      glm::mat4 view = getView();
      glm::mat4 camera_matrix = projection * view;
      py::array_t<float> c_matrix = py::array_t<float>(16);
      auto c_matrix_buf = c_matrix.request();
      float *c_ptr = (float *) c_matrix_buf.ptr;
      for (size_t i=0; i<4; ++i) {
        for (size_t j=0; j<4; ++j) {
          c_ptr[i*4 + j] = camera_matrix[j][i];
        }
      }
      c_matrix.resize({4,4});
      // return projection * view;
      return c_matrix;
    }
    py::array_t<float> getExtrinsicsNumpy() const {
      glm::mat4 camera_matrix = getView();
      py::array_t<float> c_matrix = py::array_t<float>(16);
      auto c_matrix_buf = c_matrix.request();
      float *c_ptr = (float *) c_matrix_buf.ptr;
      for (size_t i=0; i<4; ++i) {
        for (size_t j=0; j<4; ++j) {
          c_ptr[i*4 + j] = camera_matrix[j][i];
        }
      }
      c_matrix.resize({4,4});
      return c_matrix;
    }

  protected:
    void computeFront() {
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(front);
    }

};

class CameraController {
  public:
    CameraController(GLFWwindow& window, Camera& cam);

    GLFWwindow& getWindow() const { return window_; }

    // move camera pos based on keyboard states
    void moveCamera();

  private:
    // keyboard movement speed
    float key_speed_ = 6.5f;
    // mouse sensitivity
    float mouse_speed_ = 0.03f;

    glm::vec2 mouse_;
    bool mouse_init_ = false;

    GLFWwindow& window_;
    Camera& cam_;

    Timer timer_;
    float last_time_;
    // key states. pressed or not
    std::vector<bool> keys_;

    void keyCallback(int key, int /* scancode */, int action, int /* mode*/);
    void mouseCallback(double x, double y);

};

} // namespace render
