// Copyright (C) 2026 Miguel Ángel González Santamarta
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>

#include "omni_plan_llm/llm_planner.hpp"

// ---------------------------------------------------------------------------
// Grammar builder tests (no ROS / LLM required)
// ---------------------------------------------------------------------------

class LlmPlannerGrammarTest : public ::testing::Test {
protected:
  // Expose the private helper via a thin wrapper
  omni_plan_llm::LlmPlanner planner_;
};

TEST_F(LlmPlannerGrammarTest, HasSolutionReturnsFalseForEmptyString) {
  EXPECT_FALSE(planner_.has_solution(""));
}

TEST_F(LlmPlannerGrammarTest, HasSolutionReturnsFalseForPlainText) {
  EXPECT_FALSE(planner_.has_solution("No plan found"));
}

TEST_F(LlmPlannerGrammarTest, HasSolutionReturnsTrueForValidPlanLine) {
  const std::string plan = "0.000: (move r1 l1 l2)  [10.000]\n"
                           "10.000: (pick_up r1 o1 l2)  [10.000]";
  EXPECT_TRUE(planner_.has_solution(plan));
}

TEST_F(LlmPlannerGrammarTest, HasSolutionReturnsFalseForMissingBrackets) {
  // No timing brackets
  EXPECT_FALSE(planner_.has_solution("0.000: (move r1 l1 l2)"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
