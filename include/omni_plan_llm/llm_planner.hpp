// Copyright (C) 2025 Miguel Ángel González Santamarta
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

#ifndef OMNI_PLAN_LLM__LLM_PLANNER_HPP_
#define OMNI_PLAN_LLM__LLM_PLANNER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "llama_msgs/action/generate_chat_completions.hpp"
#include "llama_msgs/msg/chat_message.hpp"

#include "omni_plan/planner.hpp"

namespace omni_plan_llm {

/**
 * @class LlmPlanner
 * @brief Planner implementation that uses a large language model (LLM) via
 * llama_ros to solve PDDL planning problems.
 *
 * @details This planner performs two sequential LLM calls to generate a plan:
 *   1. A **reasoning call** that prompts the model to think through the PDDL
 *      domain and problem step-by-step, without any grammar constraint.
 *   2. A **plan-generation call** that uses the reasoning output as context
 *      and applies a GBNF grammar — built from the loaded actions and problem
 *      objects — to constrain the LLM output to syntactically valid PDDL plan
 *      steps.
 *
 * The GBNF grammar is generated dynamically for each planning request so that
 * only the action names present in the domain and only the object names
 * present in the problem are allowed in the plan output.
 */
class LlmPlanner : public omni_plan::Planner {
public:
  /**
   * @brief Default constructor for LlmPlanner.
   * @details Registers all ROS 2 parameters required by the LLM planner.
   */
  LlmPlanner();

  /**
   * @brief Generates a PDDL plan by calling an LLM via llama_ros.
   * @details Reads the domain and problem from the provided file paths,
   * constructs a GBNF grammar from the declared actions and objects, then
   * performs two sequential LLM calls: one for reasoning and one for
   * constrained plan generation.
   * @param domain The PDDL domain.
   * @param problem The PDDL problem.
   * @return A PDDL plan.
   */
  omni_plan::pddl::Plan
  generate_plan(const omni_plan::pddl::Domain &domain,
                const omni_plan::pddl::Problem &problem) const override;

  using omni_plan::Planner::generate_plan;

  /**
   * @brief Checks whether the plan output contains at least one valid action
   * step.
   * @param plan_output The raw plan string returned by generate_plan.
   * @return True if the string contains parenthesised action invocations with
   *         PDDL timing brackets, false otherwise.
   */
  bool has_solution(const std::string &plan_output) const override;

private:
  /// Name of the ROS 2 action server used for LLM inference.
  std::string llm_action_name_;
  /// Sampling temperature passed to the LLM (0.0 = deterministic).
  float temperature_;
  /// Timeout (seconds) for waiting for the LLM action server.
  int server_timeout_;

  /// Dedicated ROS 2 node used to host the action client.
  mutable std::shared_ptr<rclcpp::Node> llm_node_;
  /// Single-threaded executor used to drive the action client.
  mutable std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  /// llama_ros action client.
  mutable std::shared_ptr<
      rclcpp_action::Client<llama_msgs::action::GenerateChatCompletions>>
      llm_client_;

  /**
   * @brief Lazily initialises the ROS 2 node, executor, and action client.
   * @details Called automatically before any LLM request. Thread-unsafe but
   *          acceptable because generate_plan is always called sequentially.
   */
  void ensure_llm_client() const;

  /**
   * @brief Sends a prompt to the LLM and waits for the complete response.
   * @param prompt The text prompt to send.
   * @param grammar GBNF grammar string to constrain sampling (empty = no
   *                constraint).
   * @return The LLM response text, or an empty string on failure.
   */
  std::string call_llm(const std::string &prompt,
                       const std::string &system_prompt,
                       const std::string &grammar) const;

  /**
   * @brief Holds the parsed information for a single PDDL action.
   */
  struct ActionInfo {
    /// Action name as it appears in the PDDL domain.
    std::string name;
    /// Parameter list as (parameter_name, type) pairs.
    std::vector<std::pair<std::string, std::string>> params;
  };

  /**
   * @brief Parses the (:durative-action …) blocks from a PDDL domain string.
   * @param domain The PDDL domain.
   * @return A vector of ActionInfo structs, one per action.
   */
  std::vector<ActionInfo>
  parse_domain_actions(const omni_plan::pddl::Domain &domain) const;

  /**
   * @brief Parses the (:objects …) section from a PDDL problem string.
   * @param problem The PDDL problem.
   * @return A map from type name to the list of object names of that type.
   */
  std::unordered_map<std::string, std::vector<std::string>>
  parse_problem_objects(const omni_plan::pddl::Problem &problem) const;

  /**
   * @brief Builds a GBNF grammar that constrains the LLM to produce valid
   *        PDDL temporal plan steps.
   * @details The grammar allows only action names from @p actions and only
   *          object names drawn from @p objects_by_type for the corresponding
   *          parameter types.  Actions whose parameter types are missing from
   *          @p objects_by_type are excluded from the grammar.
   * @param actions Parsed action metadata from the PDDL domain.
   * @param objects_by_type Map of type name → list of object names from the
   *                        PDDL problem.
   * @return The GBNF grammar string, or an empty string if no valid actions
   *         remain.
   */
  std::string build_gbnf_grammar(
      const std::vector<ActionInfo> &actions,
      const std::unordered_map<std::string, std::vector<std::string>>
          &objects_by_type) const;
};

} // namespace omni_plan_llm

#endif // OMNI_PLAN_LLM__LLM_PLANNER_HPP_
