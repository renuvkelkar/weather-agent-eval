"""Weather agent that must look up conditions before it answers.

The model is a local policy so `adk eval` can run without a Gemini API key.
It still has to call `get_current_weather` and then speak from that result,
which is what the golden trajectory checks.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

_WEATHER = {
    "london": {"temperature_c": 18, "condition": "partly cloudy"},
}


def get_current_weather(location: str) -> dict:
  """Look up the current weather for a city.

  Call this tool before answering any weather question.

  Args:
    location: City name exactly as the user wrote it, for example London.

  Returns:
    A status payload whose report field is the sentence to show the user.
  """
  reading = _WEATHER.get(location.strip().lower())
  if reading is None:
    return {
        "status": "error",
        "error_message": (
            f"Weather information for '{location}' is not available."
        ),
    }

  report = (
      f"The weather in {location} today is {reading['temperature_c']}°C"
      f" and {reading['condition']}."
  )
  return {
      "status": "success",
      "location": location,
      "temperature_c": reading["temperature_c"],
      "condition": reading["condition"],
      "report": report,
  }


_LOCATION_STOPWORDS = {"what", "the", "is", "like", "in", "today"}


def _location_from_request(llm_request: LlmRequest) -> str:
  """Return the city name using the capitalization in the user text."""
  for content in llm_request.contents:
    for part in content.parts or []:
      if not part.text:
        continue
      for match in re.finditer(
          r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", part.text
      ):
        city = match.group(1)
        if city.lower() not in _LOCATION_STOPWORDS:
          return city
  return "London"


def _report_from_tool_response(llm_request: LlmRequest) -> str | None:
  """Read the weather tool result already present in this turn."""
  for content in reversed(llm_request.contents):
    for part in content.parts or []:
      function_response = part.function_response
      if (
          function_response is None
          or function_response.name != "get_current_weather"
      ):
        continue
      payload = function_response.response or {}
      if not isinstance(payload, dict):
        return str(payload)
      report = payload.get("report")
      if isinstance(report, str):
        return report
      error_message = payload.get("error_message")
      if isinstance(error_message, str):
        return error_message
  return None


class WeatherLookupModel(BaseLlm):
  """Calls the weather tool, then repeats the tool's report."""

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    report = _report_from_tool_response(llm_request)
    if report is None:
      part = types.Part(
          function_call=types.FunctionCall(
              name="get_current_weather",
              args={"location": _location_from_request(llm_request)},
          )
      )
    else:
      part = types.Part(text=report)

    yield LlmResponse(
        content=types.Content(role="model", parts=[part]),
    )


root_agent = Agent(
    name="weather_time_agent",
    model=WeatherLookupModel(model="weather-lookup-policy"),
    description=(
        "Agent that answers weather questions by looking up current conditions."
    ),
    instruction=(
        "You are a weather agent. Before you answer any weather question, "
        "call get_current_weather exactly once. Pass location as the city "
        "name only, with the same capitalization the user used. After the "
        "tool returns, reply with the report field verbatim and nothing else."
    ),
    tools=[get_current_weather],
)
