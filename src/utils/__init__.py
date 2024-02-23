def extract_assistant_output(output: str, templates: tuple[str]) -> str:
    return output[output.rindex(templates[0])+len(templates[0]):output.rindex(templates[1])].strip()
