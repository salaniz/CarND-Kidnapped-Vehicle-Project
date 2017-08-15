/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position (based on estimates of
  // x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.

  // set the number of particles
  num_particles = 50;

  // create normal distributions for x, y and theta
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // initialize the particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
    weights.push_back(1);
  }

  // set initialization variable
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.

  // create normal distributions for x, y and theta
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // iterate over particles
  for (int i = 0; i < num_particles; i++) {
    Particle &p = particles[i];
    // update particles x and y coordinates
    if (fabs(yaw_rate) < 0.001) {  // if yaw rate is zero
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    } else {  // if yaw rate is not zero
      p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
    }
    // add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);

    // update yaw
    p.theta += yaw_rate * delta_t;
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution.
  // The observations are given in the VEHICLE'S coordinate system.
  // Particles are located according to the MAP'S coordinate system.

  // calculate normalization and variance terms for Gaussian distribution
  double gauss_norm = 1. / (2. * M_PI * std_landmark[0] * std_landmark[1]);
  double var_x = 2. * std_landmark[0] * std_landmark[0];
  double var_y = 2. * std_landmark[1] * std_landmark[1];

  // iterate over particles
  for (int i = 0; i < num_particles; i++) {
    Particle &p = particles[i];
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    // avoid repeated calculations
    double sin_theta = sin(p.theta);
    double cos_theta = cos(p.theta);

    double new_weight = 1.;
    // iterate over observations
    for (LandmarkObs obs : observations) {
      // transforms observation into MAP system with respect to particle location and orientation
      double trans_x = p.x + obs.x * cos_theta - obs.y * sin_theta;
      double trans_y = p.y + obs.x * sin_theta + obs.y * cos_theta;

      Map::single_landmark_s min_landm;
      double min_dist = -1;
      // iterate over landmarks and find closest to current observation
      for (Map::single_landmark_s landm : map_landmarks.landmark_list) {
        double dst = dist(landm.x_f, landm.y_f, trans_x, trans_y);
        if (dst <= sensor_range && (dst < min_dist || min_dist == -1)) {
          min_landm = landm;
          min_dist = dst;
        }
      }

      // probability of observation for being the closest landmark:
      // calculate exponent of Gaussian distribution
      double x_dist = trans_x - min_landm.x_f;
      double y_dist = trans_y - min_landm.y_f;
      double exponent = (x_dist * x_dist) / var_x + (y_dist * y_dist) / var_y;

      // calculate weight of observation using normalization terms and exponent
      double w = gauss_norm * exp(-exponent);

      // multiply weight of current observation with all others
      new_weight *= w;
      // add debug information
      associations.push_back(min_landm.id_i);
      sense_x.push_back(trans_x);
      sense_y.push_back(trans_y);
    }

    // update weight of particle
    p.weight = new_weight;
    weights[i] = new_weight;
    // add debug information
    SetAssociations(p, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.

  // create discrete distribution with weights of particles
  default_random_engine gen;
  discrete_distribution<> resampler(weights.begin(), weights.end());

  // resample
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[resampler(gen)]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     std::vector<int> associations,
                                     std::vector<double> sense_x,
                                     std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
