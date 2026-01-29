class PromptFormatter:
    def __init__(self, system_template, human_template):
        self.system_template = system_template
        self.human_template = human_template

    def __call__(self, question, context):
        system_part = self.system_template.format(context=context)
        human_part = self.human_template.format(question=question)
        return f"{system_part}\n{human_part}"

def build_prompt(settings):
    return PromptFormatter(
        settings.prompts.extractor.system,
        settings.prompts.extractor.human,
    )
