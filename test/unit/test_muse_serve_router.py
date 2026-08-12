import unittest
from tinygrad.llm.serve import MuseStreamRouter, StreamRouter

def _feed(router, s:str, chunk:int|None=None):
  out = []
  if chunk is None:
    out.extend(router.route(s))
  else:
    for i in range(0, len(s), chunk):
      out.extend(router.route(s[i:i+chunk]))
  out.extend(router.route("", final=True))
  return out

def _join(events, field):
  return "".join(t for f, t in events if f == field)

class TestMuseStreamRouter(unittest.TestCase):
  SAMPLE = " to=self<|message|>CoT here<|eom|><|start|>assistant to=user<|message|>Hi<|eot|>"

  def test_self_then_user(self):
    ev = _feed(MuseStreamRouter(), self.SAMPLE)
    self.assertEqual(_join(ev, "reasoning_content"), "CoT here")
    self.assertEqual(_join(ev, "content"), "Hi")
    self.assertFalse(any("<|" in t or "to=" in t for _, t in ev))

  def test_chunked_one_char(self):
    ev = _feed(MuseStreamRouter(), self.SAMPLE, chunk=1)
    self.assertEqual(_join(ev, "reasoning_content"), "CoT here")
    self.assertEqual(_join(ev, "content"), "Hi")

  def test_chunked_odd(self):
    ev = _feed(MuseStreamRouter(), self.SAMPLE, chunk=3)
    self.assertEqual(_join(ev, "reasoning_content"), "CoT here")
    self.assertEqual(_join(ev, "content"), "Hi")

  def test_default_recipient_user(self):
    ev = _feed(MuseStreamRouter(), "<|message|>Hello<|eot|>")
    self.assertEqual(_join(ev, "content"), "Hello")
    self.assertEqual(_join(ev, "reasoning_content"), "")

  def test_eof_without_end_tag(self):
    ev = _feed(MuseStreamRouter(), " to=user<|message|>partial")
    self.assertEqual(_join(ev, "content"), "partial")

  def test_unknown_recipient_dropped(self):
    ev = _feed(MuseStreamRouter(), " to=browser.search<|message|>query<|eom|><|start|>assistant to=user<|message|>ok<|eot|>")
    self.assertEqual(_join(ev, "content"), "ok")
    self.assertEqual(_join(ev, "reasoning_content"), "")

  def test_stream_router_unchanged(self):
    ev = _feed(StreamRouter(reasoning=True), "reasoning</think>answer")
    self.assertEqual(ev, [("reasoning_content", "reasoning"), ("content", "answer")])

if __name__ == "__main__":
  unittest.main()
