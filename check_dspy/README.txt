Signatures:
    - str default type

    Question Answering
        question -> answer
        question: str -> answer: str

    Sentiment Classification
        sentence -> sentiment: bool

    Summarization
        document -> summary

    Retrieval-Augmented Question Answering
        context: list[str], question: str -> answer: str

    Multiple-Choice Question Answering with Reasoning
        question, choices: list[str] -> reasoning: str, selection: int


dspy.
    BaseModule (dspy.primitives.base_module)
        Module(BaseModule, metaclass=ProgramMeta) (dspy.primitives.module)
            SyncWrapper(Module) (dspy.utils.syncify)
            BestOfN(Module) (dspy.predict.best_of_n)
            ChainOfThought(Module) (dspy.predict.chain_of_thought)
            MultiChainComparison(Module) (dspy.predict.multi_chain_comparison)
            Predict(Module, Parameter) (dspy.predict.predict)
            ProgramOfThought(Module) (dspy.predict.program_of_thought)
                CodeAct(ReAct, ProgramOfThought) (dspy.predict.code_act)
            ReAct(Module) (dspy.predict.react)
                CodeAct(ReAct, ProgramOfThought) (dspy.predict.code_act)
            Refine(Module) (dspy.predict.refine)
            RLM(Module) (dspy.predict.rlm)
            GenerateModuleInstruction(dspy.Module) (dspy.propose.grounded_proposer)
            SemanticF1(Module) (dspy.evaluate.auto_evaluation)
            CompleteAndGrounded(Module) (dspy.evaluate.auto_evaluation)
            RulesInductionProgram(dspy.Module) (dspy.teleprompt.infer_rules)
            Avatar(dspy.Module) (dspy.predict.avatar.avatar)
            SingleComponentMultiModalProposer(dspy.Module) (dspy.teleprompt.gepa.instruction_proposal)




pypi: dspy or dspy-ai - they are the same package

    it seems based on 'LiteLLM' !!!

DSPy: https://dspy.ai/learn/programming/language_models/

    openai/<name>
    gemini/...
    anthropic/...
    vertex_ai/...
    databricks/...
    ollama_chat/...,    api_base='http://localhost:11434'
    lm_studio/...,      api_base='http://127.0.0.1:1234'
    ... all LLMs supported by LiteLLM


LiteLLM:  https://docs.litellm.ai/docs/providers
