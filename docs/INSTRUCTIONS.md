# Instructions

## Initial Prompt:
Read and review the docs, namely the PLAN.md, REFERENCE.md, and RESEARCH.md to get a solid understanding of the project development plan. This is the start of the project and versioning is being kept in GitHub. Please start by doing deep research, enhancing the documentation, plan and SPEC. Work through the planning systematically and create a comprehensive task list to follow and keep track of (create a TASKS.md file, CHANGELOG.md, and any other documents you see fit).

Put documents in the ./docs/ folder. Do not clutter up the root. The only documents there should be the README.md (including links with summaries to all other docs with simple getting started instructions, etc.) and LICENSE (only because GitHub looks for it there). After you have a solid plan and all documentation, SPECS, PRD, etc. developed... then cleanup your list and start development, systematically testing and validating things as you go, developing and cleaning up task lists as you go, improve the code programatically while creating test coverage as you go.

Follow a solid SDLC framework, best practices, and methodologies defined. Work on getting things to work, then adding features, then optimizations. Continiously improve the TTS system and add the requested features. Begin immediately by reading the existing documentation files to establish baseline understanding, then proceed systematically through each phase while maintaining detailed progress tracking.

Setup priorities and phases to be completed. Analyze your work product and reflect on findings, lessions learned, etc. Keep track of useful research, memories, and etc. Continue the task list systematically until all are completed. Build a plan of action along with a comprehensive task list and work through it systematically until all tasks are completed.

The single recommended command to run the API should be `uv run python app.py` (focus on using 'uv' runtime as the Python is managed on this system and it's best practice to use uv, which takes care of most things for you, like automatically installs dependancies, etc.).

## Enhanced:
Conduct a comprehensive project analysis and planning phase for the JabberTTS project. This is a multi-phase process that must be executed systematically to establish a solid foundation for development.

**EXECUTION ORDER (CRITICAL - Follow This Exact Sequence):**

**Phase 1: Discovery and Analysis**
1. **Documentation Review** - Use `view` tool to examine ALL existing documentation:
   - PLAN.md, REFERENCE.md, RESEARCH.md (if they exist)
   - README.md files (root and subdirectories)
   - All .md files in project root
   - Document what exists, what's missing, and what's outdated

2. **Codebase Analysis** - Use `codebase-retrieval` tool to understand:
   - Project structure and file organization
   - Current implementation patterns and architecture
   - Technology stack and dependencies
   - Configuration files and build systems
   - Code quality and technical debt

3. **Development History** - Use `git-commit-retrieval` tool to analyze:
   - Recent development activity and patterns
   - Architectural decisions and their rationale
   - Failed experiments or abandoned approaches
   - Evolution of project scope and goals

4. **Industry Research** - Use `web-search` tool to research:
   - Current TTS technology best practices (2024-2025)
   - Popular frameworks: Coqui TTS, Mozilla TTS, Tacotron
   - Performance benchmarks and quality metrics
   - Modern deployment and integration patterns

**Phase 2: Task Planning (BEFORE Documentation)**
1. **Create Initial Task Structure** - Use `add_tasks` to establish:
   - High-level phases: Analysis → Documentation → Development → Testing
   - Each task should represent ~20 minutes of focused work
   - Clear parent-child relationships and dependencies
   - Priority levels: P0 (Critical/MVP) → P1 (High) → P2 (Medium) → P3 (Low)

2. **Task State Management** - Use `update_tasks` to:
   - Mark discovery tasks as IN_PROGRESS/COMPLETE as you work
   - Maintain accurate progress tracking throughout

**Phase 3: Comprehensive Documentation Creation**
Create `./docs/` directory structure with these specific files using `save-file`:

**Core Planning Documents:**
- **PLAN.md**: Development roadmap with weekly milestones, success criteria, risk mitigation
- **SPEC.md**: Functional/technical specifications, performance requirements, API contracts
- **PRD.md**: Product Requirements with user personas, use cases, acceptance criteria

**Technical Documentation:**
- **REFERENCE.md**: Architecture diagrams, API docs, configuration, troubleshooting
- **RESEARCH.md**: Technology comparisons, ADRs (Architectural Decision Records)
- **TESTING.md**: Testing strategy, coverage requirements (minimum 80%), test scenarios

**Project Management:**
- **TASKS.md**: Hierarchical breakdown with dependencies, estimates, priorities
- **CHANGELOG.md**: Version history with semantic versioning
- **CONTRIBUTING.md**: Development guidelines, code standards, PR process

**Memory and Learning Integration:**
- Use the `remember` tool to capture key architectural decisions, technology evaluations, and lessons learned
- Document research findings, technology comparisons, and evaluation criteria for future reference
- Track successful development patterns and approaches for reuse in future project phases
- Maintain comprehensive decision rationale for future team members and project evolution

**Phase 4: Project Structure Updates**
1. **Update Root README.md** - Use `str-replace-editor` to include:
   - Clear project description and value proposition
   - Quick start guide with installation steps
   - Links to ./docs/ with descriptions
   - Development setup using `uv run python app.py`
   - Build status and coverage badges

2. **Verify LICENSE** - Ensure compliance documentation exists

**Phase 5: Development Strategy Implementation**
1. **MVP Focus**: Core TTS functionality first
   - Text input processing and validation
   - Audio output generation
   - Basic configuration and error handling

2. **Incremental Development**: Feature additions by priority
   - Voice customization options
   - Advanced audio processing
   - Performance optimizations

3. **Quality Assurance**: Use `launch-process` for:
   - Test execution: `uv run python -m pytest`
   - Performance benchmarking
   - Integration testing

**CRITICAL REQUIREMENTS:**
- **Tool Usage**: Use ALL specified tools systematically (view, codebase-retrieval, git-commit-retrieval, save-file, str-replace-editor, task management)
- **Python Runtime**: Always use `uv run python app.py` for execution (uv handles dependencies automatically)
- **Testing Strategy**: Implement comprehensive testing with `launch-process` at each milestone
- **Progress Tracking**: Update task states regularly using `update_tasks`
- **Memory Integration**: Use `remember` tool for key decisions and lessons learned

**SUCCESS CRITERIA (Measurable):**
- Complete documentation suite in ./docs/ with consistent formatting
- Task management system with realistic estimates and clear dependencies
- Development strategy with specific milestones and timelines
- Project structure optimized for team collaboration
- All existing documentation analyzed with gap assessment

**EXECUTION INSTRUCTIONS:**
1. Start immediately with Phase 1 discovery
2. Do NOT skip phases or rush analysis
3. Complete task planning BEFORE creating documentation
4. Use parallel tool calls when reading multiple files
5. Update task states as you progress through each phase
6. Build comprehensive understanding before making any code changes

Begin with reading existing documentation to establish baseline understanding, then proceed systematically through each phase while maintaining detailed progress tracking.
