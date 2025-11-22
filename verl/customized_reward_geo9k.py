# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from mathruler.grader import extract_boxed_content, grade_answer
import warnings
import math

# def is_close_enough(t_input, b_input):
#     if t_input == "None" or b_input == "None":
#         return False

#     try:
#         if isinstance(t_input, str):
#             # Step 1: Replace \sqrt{...} with math.sqrt(...)
#             expr = re.sub(r'\\sqrt{([^}]+)}', r'math.sqrt(\1)', t_input)

#             # Step 2: Insert * between number and math.sqrt
#             expr = re.sub(r'(\d)(math\.sqrt)', r'\1*\2', expr)

#             # Step 3: Insert * between number and opening parenthesis, e.g., 162( → 162*(
#             expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)

#             # Step 4: Insert * between closing parenthesis and number or variable, e.g., )( → )*(
#             expr = re.sub(r'\)(\s*\d)', r')*\1', expr)
#             expr = re.sub(r'\)(\s*math\.sqrt)', r')*\1', expr)

#             # Catch SyntaxWarnings as exceptions
#             with warnings.catch_warnings():
#                 warnings.simplefilter("error", SyntaxWarning)
#                 t = eval(expr, {"__builtins__": {}}, {"math": math})
#         else:
#             t = float(t_input)

#         b = float(b_input)
#     except Warning as w:
#         # print(f"SyntaxWarning during evaluation: {w}")
#         # print(f"Expression: {t_input}")
#         return False
#     except Exception:
#         return False

#     if t == b == 0:
#         return True
#     if b == 0:
#         return False

#     relative_error = abs(t - b) / abs(b)
#     return relative_error <= 0.01  # Within 1%

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r'<think>.*</think>.*\\boxed\{.*\}.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if (grade_answer(answer, ground_truth)) else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    return 0.9 * acc_reward(solution_str, ground_truth) + 0.1 * format_reward(solution_str)
