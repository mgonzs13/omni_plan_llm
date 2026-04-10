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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "llama_msgs/msg/chat_message.hpp"

#include "omni_plan_llm/llm_planner.hpp"

using namespace omni_plan_llm;
using GenerateChatCompletions = llama_msgs::action::GenerateChatCompletions;

LlmPlanner::LlmPlanner() : omni_plan::Planner() {
  this->add_ros_parameters(
      {{"llm_action_name", std::string("/llama/generate_chat_completions"),
        this->llm_action_name_},
       {"temperature", 0.2f, this->temperature_},
       {"server_timeout", 30, this->server_timeout_}});
}

void LlmPlanner::ensure_llm_client() const {
  if (this->llm_node_) {
    return;
  }

  this->llm_node_ = rclcpp::Node::make_shared(
      "llm_planner_client", rclcpp::NodeOptions().use_global_arguments(false));
  this->executor_ =
      std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  this->executor_->add_node(this->llm_node_);
  this->llm_client_ = rclcpp_action::create_client<GenerateChatCompletions>(
      this->llm_node_, this->llm_action_name_);
}

std::string LlmPlanner::call_llm(const std::string &prompt,
                                 const std::string &system_prompt,
                                 const std::string &grammar) const {
  this->ensure_llm_client();

  // Wait for the action server
  if (!this->llm_client_->wait_for_action_server(
          std::chrono::seconds(this->server_timeout_))) {
    RCLCPP_ERROR(this->llm_node_->get_logger(),
                 "LLM action server '%s' not available after %d s",
                 this->llm_action_name_.c_str(), this->server_timeout_);
    return "";
  }

  // Build the goal with chat messages
  auto goal = GenerateChatCompletions::Goal();

  // Optional system prompt
  if (!system_prompt.empty()) {
    llama_msgs::msg::ChatMessage system_msg;
    system_msg.role = "system";
    system_msg.content = system_prompt;
    goal.messages.push_back(system_msg);
  }

  // Create a user message with the prompt
  llama_msgs::msg::ChatMessage user_msg;
  user_msg.role = "user";
  user_msg.content = prompt;
  goal.messages.push_back(user_msg);

  goal.sampling_config.temp = this->temperature_;
  goal.sampling_config.grammar = grammar;

  // Send goal
  auto send_future = this->llm_client_->async_send_goal(goal);
  auto send_status = this->executor_->spin_until_future_complete(send_future);
  if (send_status != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(this->llm_node_->get_logger(),
                 "Failed to send goal to LLM action server");
    return "";
  }

  auto goal_handle = send_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(this->llm_node_->get_logger(),
                 "LLM action server rejected goal");
    return "";
  }

  // Wait for result
  auto result_future = this->llm_client_->async_get_result(goal_handle);
  auto result_status =
      this->executor_->spin_until_future_complete(result_future);
  if (result_status != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(this->llm_node_->get_logger(),
                 "Failed to get result from LLM action server");
    return "";
  }

  auto wrapped = result_future.get();
  if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_ERROR(this->llm_node_->get_logger(),
                 "LLM action finished with non-success result code %d",
                 static_cast<int>(wrapped.code));
    return "";
  }

  // Extract the text from the first chat choice
  if (wrapped.result->choices.empty()) {
    RCLCPP_ERROR(this->llm_node_->get_logger(),
                 "LLM returned empty chat choices");
    return "";
  }

  return wrapped.result->choices[0].message.content;
}

std::vector<LlmPlanner::ActionInfo>
LlmPlanner::parse_domain_actions(const std::string &domain_str) const {
  std::vector<ActionInfo> actions;

  // Match (:durative-action NAME … :parameters (…) …)
  // We iterate over all occurrences of ":durative-action"
  std::regex action_block_re(
      R"(\(:durative-action\s+([\w-]+)[\s\S]*?:parameters\s*\(([^)]*)\))",
      std::regex::icase);

  auto begin = std::sregex_iterator(domain_str.begin(), domain_str.end(),
                                    action_block_re);
  auto end = std::sregex_iterator();

  for (auto it = begin; it != end; ++it) {
    const std::smatch &match = *it;
    ActionInfo info;
    info.name = match[1].str();

    // Parse parameter pairs: ?name - type
    std::string params_str = match[2].str();
    std::regex param_re(R"(\?\s*([\w-]+)\s+-\s+([\w-]+))");
    auto pb =
        std::sregex_iterator(params_str.begin(), params_str.end(), param_re);
    auto pe = std::sregex_iterator();
    for (auto pi = pb; pi != pe; ++pi) {
      const std::smatch &pm = *pi;
      info.params.emplace_back(pm[1].str(), pm[2].str());
    }

    actions.push_back(std::move(info));
  }

  return actions;
}

std::unordered_map<std::string, std::vector<std::string>>
LlmPlanner::parse_problem_objects(const std::string &problem_str) const {
  std::unordered_map<std::string, std::vector<std::string>> objects_by_type;

  // Extract the content of (:objects … )
  std::regex objects_block_re(R"(\(:objects([\s\S]*?)\))", std::regex::icase);
  std::smatch block_match;
  if (!std::regex_search(problem_str, block_match, objects_block_re)) {
    return objects_by_type;
  }

  std::string objects_str = block_match[1].str();

  // Each line: name - type  (multiple names per type on one line are supported
  // by the omni_plan format which places one object per line)
  std::regex obj_re(R"(\b([\w-]+)\s+-\s+([\w-]+))");
  auto ob =
      std::sregex_iterator(objects_str.begin(), objects_str.end(), obj_re);
  auto oe = std::sregex_iterator();
  for (auto oi = ob; oi != oe; ++oi) {
    const std::smatch &om = *oi;
    const std::string &obj_name = om[1].str();
    const std::string &type_name = om[2].str();
    objects_by_type[type_name].push_back(obj_name);
  }

  return objects_by_type;
}

std::string LlmPlanner::build_gbnf_grammar(
    const std::vector<ActionInfo> &actions,
    const std::unordered_map<std::string, std::vector<std::string>>
        &objects_by_type) const {

  // Sanitise a name for use as a GBNF rule identifier
  // (keep only alphanumeric and '-', replace underscores and other chars with
  // '-')
  auto sanitise = [](const std::string &s) -> std::string {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
      result +=
          (std::isalnum(static_cast<unsigned char>(c)) || c == '-') ? c : '-';
    }
    return result;
  };

  // Filter actions: only keep those whose parameter types all have objects
  std::vector<const ActionInfo *> valid_actions;
  for (const auto &action : actions) {
    bool all_types_ok = true;
    for (const auto &param : action.params) {
      if (objects_by_type.find(param.second) == objects_by_type.end() ||
          objects_by_type.at(param.second).empty()) {
        all_types_ok = false;
        RCLCPP_WARN(rclcpp::get_logger("LlmPlanner"),
                    "Action '%s' skipped: type '%s' has no objects",
                    action.name.c_str(), param.second.c_str());
        break;
      }
    }
    if (all_types_ok) {
      valid_actions.push_back(&action);
    }
  }

  if (valid_actions.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("LlmPlanner"),
                "No valid actions found for GBNF grammar generation");
    return "";
  }

  // Collect only the types actually referenced by valid actions
  std::unordered_map<std::string, std::vector<std::string>> used_types;
  for (const auto *action : valid_actions) {
    for (const auto &param : action->params) {
      if (used_types.find(param.second) == used_types.end()) {
        used_types[param.second] = objects_by_type.at(param.second);
      }
    }
  }

  std::string g;

  // root and plan structure
  g += "root   ::= plan\n";
  g += "plan   ::= step (\"\\n\" step)*\n";
  // step: <float>: (<action>)  [<float>]
  g += "step   ::= number \": (\" action \")  [\" number \"]\"\n";
  g += "number ::= [0-9]+ (\".\" [0-9]+)?\n";
  g += "\n";

  // action rule: union of all valid action rules
  g += "action ::= ";
  for (size_t i = 0; i < valid_actions.size(); ++i) {
    g += "action-" + sanitise(valid_actions[i]->name);
    if (i < valid_actions.size() - 1) {
      g += " | ";
    }
  }
  g += "\n";
  g += "\n";

  // Individual action rules
  for (const auto *action : valid_actions) {
    g += "action-" + sanitise(action->name) + " ::= \"" + action->name + "\"";
    for (const auto &param : action->params) {
      g += " \" \" type-" + sanitise(param.second);
    }
    g += "\n";
  }
  g += "\n";

  // Type rules (objects)
  for (const auto &[type_name, objs] : used_types) {
    g += "type-" + sanitise(type_name) + " ::= ";
    for (size_t i = 0; i < objs.size(); ++i) {
      g += "\"" + objs[i] + "\"";
      if (i < objs.size() - 1) {
        g += " | ";
      }
    }
    g += "\n";
  }

  return g;
}

std::string LlmPlanner::generate_plan(const std::string domain_path,
                                      const std::string problem_path) const {

  // ── 1. Read domain and problem files ──────────────────────────────────────
  auto read_file = [](const std::string &path) -> std::string {
    std::ifstream f(path);
    if (!f.is_open()) {
      return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
  };

  const std::string domain_str = read_file(domain_path);
  const std::string problem_str = read_file(problem_path);

  if (domain_str.empty() || problem_str.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("LlmPlanner"),
                 "Failed to read domain ('%s') or problem ('%s') file",
                 domain_path.c_str(), problem_path.c_str());
    return "";
  }

  // ── 2. Build GBNF grammar from parsed actions and objects ─────────────────
  const auto actions = this->parse_domain_actions(domain_str);
  const auto objects_by_type = this->parse_problem_objects(problem_str);
  const std::string grammar =
      this->build_gbnf_grammar(actions, objects_by_type);

  if (grammar.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("LlmPlanner"),
                 "Could not build a valid GBNF grammar for the current "
                 "domain/problem; aborting LLM planning");
    return "";
  }

  RCLCPP_INFO(rclcpp::get_logger("LlmPlanner"), "GBNF grammar:\n%s",
              grammar.c_str());

  // ── 3. First LLM call: generate plan ─────────────────────────────────────
  std::string system_prompt_planning =
      "You are a PDDL planning expert. Generate a sequence of actions to solve "
      "the problem, ordering them logically.";

  std::string plan_prompt = "Domain:\n" + domain_str + "\n\nProblem:\n" +
                            problem_str +
                            "\n\nGenerate an ordered sequence of actions to "
                            "solve this problem.";

  RCLCPP_INFO(rclcpp::get_logger("LlmPlanner"),
              "LLM planning — step 1: plan generation");
  const std::string plan =
      this->call_llm(plan_prompt, system_prompt_planning, "");

  if (plan.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("LlmPlanner"),
                "LLM plan call returned an empty response");
    return "";
  }
  RCLCPP_DEBUG(rclcpp::get_logger("LlmPlanner"), "Generated plan:\n%s",
               plan.c_str());

  // ── 4. Second LLM call: format as PDDL temporal plan ──────────────────────
  std::string system_prompt_format =
      "Convert the action sequence into PDDL temporal plan format. Each line "
      "must be: <time>: (<action> <params>) [<duration>]";

  std::string format_prompt = "Action sequence:\n" + plan +
                              "\n\nConvert this to PDDL temporal plan format, "
                              "one action per line, with "
                              "no explanation.";

  RCLCPP_INFO(rclcpp::get_logger("LlmPlanner"),
              "LLM planning — step 2: PDDL format conversion");
  const std::string pddl_plan =
      this->call_llm(format_prompt, system_prompt_format, grammar);

  if (pddl_plan.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("LlmPlanner"),
                "LLM format conversion returned an empty response");
  } else {
    RCLCPP_DEBUG(rclcpp::get_logger("LlmPlanner"), "PDDL plan:\n%s",
                 pddl_plan.c_str());
  }

  return pddl_plan;
}

bool LlmPlanner::has_solution(const std::string &plan_output) const {
  // A plan is considered valid when the output contains at least one PDDL
  // temporal action step, i.e. a line with both parentheses and timing
  // brackets: "<time>: (<action> ...) [<duration>]"
  if (plan_output.empty()) {
    return false;
  }
  std::istringstream ss(plan_output);
  std::string line;
  while (std::getline(ss, line)) {
    if (line.find('(') != std::string::npos &&
        line.find(')') != std::string::npos &&
        line.find('[') != std::string::npos &&
        line.find(']') != std::string::npos) {
      return true;
    }
  }
  return false;
}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(omni_plan_llm::LlmPlanner, omni_plan::Planner)
