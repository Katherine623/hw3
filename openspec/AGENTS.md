# OpenSpec Agent Workflow

## Overview
This document describes how to work with AI coding assistants (like GitHub Copilot) using the OpenSpec spec-driven development workflow.

## Workflow Steps

### 1. Project Context Setup
**Purpose:** Establish the foundation for all future interactions.

**Steps:**
1. Read `openspec/project.md` thoroughly
2. Understand the tech stack, conventions, and project structure
3. Confirm understanding with the AI assistant

**Example Prompt:**
```
Please read openspec/project.md and confirm you understand 
the project context, tech stack, and conventions.
```

### 2. Feature Development Cycle

#### 2.1 Propose Changes
Before implementing any new feature, create a change proposal.

**Example Prompt:**
```
I want to add [FEATURE NAME]. Please create an OpenSpec 
change proposal for this feature.
```

**Proposal Template:**
```markdown
## Change Proposal: [Feature Name]

### Motivation
Why is this change needed?

### Proposed Changes
- File: [filename]
  - Add/Modify: [description]
- File: [filename]
  - Add/Modify: [description]

### Testing Plan
How will we verify this works?

### Documentation Updates
What docs need updating?
```

#### 2.2 Implementation
Once the proposal is approved, implement the changes.

**Example Prompt:**
```
Let's implement the approved proposal for [FEATURE NAME].
Follow the plan in openspec/proposals/[proposal-file].md
```

#### 2.3 Verification
Test and verify the implementation.

**Example Prompt:**
```
Please help me test the [FEATURE NAME] implementation 
and verify it meets the requirements.
```

### 3. Iteration and Refinement

**Feedback Loop:**
1. Test the implementation
2. Identify issues or improvements
3. Create new proposals for refinements
4. Implement and verify

**Example Prompt:**
```
The [FEATURE] works but has [ISSUE]. Please propose 
improvements to address this.
```

## Best Practices

### For Students (Developers)
1. **Always read project.md first** before making requests
2. **Be specific** in your prompts
3. **Reference the project context** when asking questions
4. **Review proposals** before implementation
5. **Test incrementally** after each change
6. **Document as you go** - update README when features are added

### For AI Assistants
1. **Refer to project.md** for conventions and context
2. **Create proposals** before major changes
3. **Explain your reasoning** when suggesting alternatives
4. **Follow the tech stack** specified in project.md
5. **Maintain consistency** with existing code style
6. **Test suggestions** when possible

## Common Workflows

### Adding a New ML Model
```
1. Prompt: "I want to add Naïve Bayes model. Create a proposal."
2. Review the proposal
3. Prompt: "Implement the Naïve Bayes model as proposed."
4. Prompt: "Add evaluation metrics for the new model."
5. Prompt: "Update the README with the new model information."
```

### Enhancing Visualizations
```
1. Prompt: "I want to add a confusion matrix visualization. 
   Create a proposal."
2. Review the proposal
3. Prompt: "Implement the confusion matrix visualization."
4. Prompt: "Test the visualization with sample data."
```

### Fixing Deployment Issues
```
1. Prompt: "The app fails to deploy with [ERROR]. 
   What's the issue and how do we fix it?"
2. Review the suggested solution
3. Prompt: "Implement the fix for the deployment issue."
4. Prompt: "Verify the fix works locally before pushing."
```

## Communication Guidelines

### Effective Prompts
✅ **Good:** "Add a Naïve Bayes classifier following the project structure in project.md"
❌ **Bad:** "Add ML model"

✅ **Good:** "Following CRISP-DM, implement data preprocessing with text cleaning and TF-IDF"
❌ **Bad:** "Clean the data"

✅ **Good:** "Create an interactive confusion matrix using plotly, consistent with other visualizations"
❌ **Bad:** "Make a chart"

### When to Create Proposals
- **Always:** For new features, major refactoring, or architectural changes
- **Optional:** For bug fixes, minor improvements, or documentation updates
- **Never:** For trivial changes like fixing typos

## Version Control Integration

### Commit Message Format
```
[Phase] Brief description

Detailed explanation if needed.

Related to: openspec/proposals/[proposal-file].md
```

### Example Commits
```
Phase 1 – Preprocessing: Add text cleaning pipeline

Implemented lowercase conversion, punctuation removal, 
and tokenization following CRISP-DM data preparation phase.

Related to: openspec/proposals/001-text-preprocessing.md
```

```
Phase 2 – Modeling: Add Naïve Bayes classifier

Implemented MultinomialNB with hyperparameter tuning.
Added to model comparison dashboard.

Related to: openspec/proposals/002-naive-bayes.md
```

## Troubleshooting

### Issue: AI doesn't follow project conventions
**Solution:** Re-reference project.md explicitly in your prompt

### Issue: Changes break existing functionality
**Solution:** Request rollback and create a more detailed proposal

### Issue: Unclear what to do next
**Solution:** Ask "What should I work on next based on the homework requirements?"

## References
- [OpenSpec Tutorial Video](https://www.youtube.com/watch?v=ANjiJQQIBo0)
- [CRISP-DM Methodology](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- Project Requirements: See homework assignment document
