import argparse


def create_ra_parser(subparsers):
    ra_parser = subparsers.add_parser("RA", help="RA agent specific arguments")
    ra_parser.add_argument('--model', choices=[
        'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'gpt-4o-2024-05-13',
        'deepseek-coder-6.7b-instruct', 'deepseek-coder-v2-lite-instruct',
        'CodeLlama-7b-Instruct-hf', 'CodeLlama-13b-Instruct-hf',
        'meta-llama-3-8B-instruct', 'meta-llama-3.1-8b-instruct',
        'meta-llama-3-70b-instruct', 'meta-llama-3.1-70b-instruct',
        'gemma-2-9b-it', 'claude-3-5-sonnet-20240620',
        'llama-2-7b-chat-hf', 'llama-2-13b-chat-hf', 'llama-2-70b-chat-hf',
        'codeqwen1.5-7b-chat', 'gpt-35-turbo', 'gpt-5.1', 'gpt-4.1'],
                           help='The base model to use in RA function.')
    ra_parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for RA model.')
    ra_parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus sampling) for RA model.')
    ra_parser.add_argument('--seed', type=int, default=998244353, help='Random seed for RA model.')
    ra_parser.add_argument('--max_tokens', type=int, default=1024, help='Token length for RA model.')
    ra_parser.add_argument('--dry_run', type=bool, default=False)
    ra_parser.add_argument('--version', type=str, default='v1', help='Version of the dataset for RA.')
    ra_parser.add_argument('--safety_sys_choice', choices=[
        'none', '0_safety_sys', '1_safety_sys_awareness', '2_safety_sys_detection',
        '3_safety_sys_modification', '4_safety_integration'],
                           default='none', help='Safety system choice for RA.')
    group = ra_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--python_eval', action='store_true', help='Flag for Python evaluation in RA.')
    group.add_argument('--bash_eval', action='store_true', help='Flag for Bash evaluation in RA.')
    return ra_parser


def create_ra_test_parser(subparsers):
    ra_parser = subparsers.add_parser("RA_TEST", help="RA agent specific arguments")
    ra_parser.add_argument('--model', choices=[
        'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'gpt-4o-2024-05-13',
        'deepseek-coder-6.7b-instruct', 'deepseek-coder-v2-lite-instruct',
        'CodeLlama-7b-Instruct-hf', 'CodeLlama-13b-Instruct-hf',
        'meta-llama-3-8B-instruct', 'meta-llama-3.1-8b-instruct',
        'meta-llama-3-70b-instruct', 'meta-llama-3.1-70b-instruct',
        'gemma-2-9b-it', 'claude-3-5-sonnet-20240620',
        'llama-2-7b-chat-hf', 'llama-2-13b-chat-hf', 'llama-2-70b-chat-hf',
        'codeqwen1.5-7b-chat', 'gpt-35-turbo', 'gpt-5.1', 'gpt-4.1'],
                           help='The base model to use in RA function.')
    ra_parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for RA model.')
    ra_parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus sampling) for RA model.')
    ra_parser.add_argument('--seed', type=int, default=998244353, help='Random seed for RA model.')
    ra_parser.add_argument('--max_tokens', type=int, default=1024, help='Token length for RA model.')
    ra_parser.add_argument('--dry_run', type=bool, default=False)
    ra_parser.add_argument('--version', type=str, default='v1', help='Version of the dataset for RA.')
    ra_parser.add_argument('--safety_sys_choice', choices=[
        'none', '0_safety_sys', '1_safety_sys_awareness', '2_safety_sys_detection',
        '3_safety_sys_modification', '4_safety_integration'],
                           default='none', help='Safety system choice for RA.')
    group = ra_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--python_eval', action='store_true', help='Flag for Python evaluation in RA.')
    group.add_argument('--bash_eval', action='store_true', help='Flag for Bash evaluation in RA.')
    return ra_parser


def main():
    # Create top-level parser
    parser = argparse.ArgumentParser(description="Run specified model with given LLM and risky_id range")

    # Choose the agent type (RA, RA_TEST)
    subparsers = parser.add_subparsers(dest="agent", required=True)

    # Add sub-parsers for each agent type
    create_ra_parser(subparsers)
    create_ra_test_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the correct sub-parser based on the chosen agent type
    if args.agent == "RA":
        from RedCode_Exec.RA_evaluation.RA import RA
        print(f"RA selected with args: {args}")
        if args.python_eval:
            task_type = 'python_eval'
        if args.bash_eval:
            task_type = 'bash_eval'
        RA(args.model, args.temperature, args.top_p, args.seed, args.max_tokens,
           args.dry_run, args.version, args.safety_sys_choice, task_type=task_type)
    elif args.agent == "RA_TEST":
        from RedCode_Exec.RA_evaluation_test.RA import RA
        print(f"RA selected with args: {args}")
        if args.python_eval:
            task_type = 'python_eval'
        if args.bash_eval:
            task_type = 'bash_eval'
        RA(args.model, args.temperature, args.top_p, args.seed, args.max_tokens,
           args.dry_run, args.version, args.safety_sys_choice, task_type=task_type)


if __name__ == "__main__":
    main()
