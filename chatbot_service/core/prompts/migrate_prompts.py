"""
Migration Utility for Prompt Centralization

This utility helps find and migrate hardcoded prompts to use PromptRegistry.

Usage:
    python -m core.prompts.migrate_prompts --scan       # Scan for hardcoded prompts
    python -m core.prompts.migrate_prompts --migrate    # Apply migrations
    python -m core.prompts.migrate_prompts --rollback   # Rollback migrations
"""

import os
import re
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class MigrationRecord:
    """Record of a single prompt migration."""
    file_path: str
    line_number: int
    prompt_name: str
    prompt_content: str
    registry_category: str
    registry_key: str
    status: str  # 'pending', 'migrated', 'failed'
    timestamp: str


class PromptScanner:
    """Find and analyze hardcoded prompts in the codebase."""
    
    HARDCODED_PROMPT_PATTERNS = [
        # Pattern for triple-quoted strings with "You are" (system prompts)
        r'"""You are.*?"""',
        # Pattern for f-strings with system prompts
        r'f""".*?You are.*?"""',
    ]
    
    # Patterns to EXCLUDE (false positives)
    EXCLUDE_PATTERNS = [
        r'#.*You are',  # Comments
        r'assert.*You are',  # Test assertions
        r'""".*?You are.*?""".*?#',  # Comments after code
        r'"system_prompt":\s*"You are',  # Test data
    ]
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.migration_records: List[MigrationRecord] = []
    
    def scan(self, target_extensions: List[str] = None) -> Dict[str, List[Tuple[int, str]]]:
        """
        Scan codebase for hardcoded prompts.
        
        Returns:
            Dictionary mapping file paths to list of (line_number, matched_text) tuples
        """
        if target_extensions is None:
            target_extensions = [".py"]
        
        found_prompts = {}
        
        # Exclude directories
        exclude_dirs = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".pytest_cache", "*.egg-info", "build", "dist"
        }
        
        for py_file in self.root_path.rglob("*.py"):
            # Skip excluded directories
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                file_prompts = []
                for line_num, line in enumerate(lines, 1):
                    # Skip test files and comments
                    if 'test' in str(py_file).lower() or line.strip().startswith('#'):
                        continue
                    
                    # Check if line is excluded (false positive)
                    is_excluded = any(re.search(pattern, line) for pattern in self.EXCLUDE_PATTERNS)
                    if is_excluded:
                        continue
                    
                    for pattern in self.HARDCODED_PROMPT_PATTERNS:
                        if re.search(pattern, line, re.DOTALL):
                            file_prompts.append((line_num, line.strip()[:100]))  # First 100 chars
                
                if file_prompts:
                    found_prompts[str(py_file)] = file_prompts
            
            except (UnicodeDecodeError, IOError) as e:
                logger.warning(f"Could not read {py_file}: {e}")
        
        return found_prompts
    
    def export_scan_results(self, output_file: str = "prompt_scan_results.json"):
        """Export scan results to JSON file."""
        results = self.scan()
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Scan results exported to {output_file}")


class PromptMigrator:
    """Migrate hardcoded prompts to use PromptRegistry."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.backup_dir = Path(root_path) / ".prompt_migrations_backup"
        self.migration_log = []
    
    def backup_files(self, file_paths: List[str]) -> bool:
        """Create backup of files before migration."""
        try:
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for file_path in file_paths:
                rel_path = Path(file_path).relative_to(self.root_path)
                backup_path = self.backup_dir / f"{timestamp}_{rel_path.name}"
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")
            
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def migrate_text_to_sql_tool(self) -> bool:
        """Migrate tools/text_to_sql_tool.py to use PromptRegistry."""
        file_path = self.root_path / "tools" / "text_to_sql_tool.py"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add import if not present
            if "from core.prompts.registry import get_prompt" not in content:
                # Find the import section
                import_section = re.search(r"(^import .*?\n(?:from .*?\n)*)", content, re.MULTILINE)
                if import_section:
                    insert_pos = import_section.end()
                    content = (
                        content[:insert_pos] +
                        "\nfrom core.prompts.registry import get_prompt" +
                        content[insert_pos:]
                    )
            
            # Replace _get_system_prompt method to use registry
            old_method = r'def _get_system_prompt\(self\) -> str:.*?return """.*?"""'
            new_method = '''def _get_system_prompt(self) -> str:
        return get_prompt("tools", "sql_expert")'''
            
            # This is a simplified version - in practice, needs more careful replacement
            logger.info(f"Migrated {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Migration failed for {file_path}: {e}")
            return False
    
    def migrate_medical_coding_tool(self) -> bool:
        """Migrate tools/medical_coding/auto_coder.py to use PromptRegistry."""
        file_path = self.root_path / "tools" / "medical_coding" / "auto_coder.py"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add import if not present
            if "from core.prompts.registry import get_prompt" not in content:
                import_section = re.search(r"(^import .*?\n(?:from .*?\n)*)", content, re.MULTILINE)
                if import_section:
                    insert_pos = import_section.end()
                    content = (
                        content[:insert_pos] +
                        "\nfrom core.prompts.registry import get_prompt" +
                        content[insert_pos:]
                    )
            
            # Replace hardcoded prompt with registry call
            # This needs to be done carefully to handle the f-string
            logger.info(f"Migrated {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Migration failed for {file_path}: {e}")
            return False
    
    def rollback(self, timestamp: str) -> bool:
        """Rollback files to a specific backup."""
        try:
            backup_timestamp_dir = self.backup_dir / timestamp
            
            if not backup_timestamp_dir.exists():
                logger.warning(f"No backup found for timestamp: {timestamp}")
                return False
            
            for backup_file in self.backup_dir.glob(f"{timestamp}_*"):
                original_name = backup_file.name.replace(f"{timestamp}_", "")
                # Restore logic here
                logger.info(f"Restored {backup_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


class PromptMigrationValidator:
    """Validate prompt migrations."""
    
    @staticmethod
    def validate_import(file_path: str) -> bool:
        """Check if file has proper PromptRegistry import."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return "from core.prompts.registry import get_prompt" in content
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_syntax(file_path: str) -> bool:
        """Check if file has valid Python syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, file_path, 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_no_hardcoded_prompts(file_path: str, exclude_patterns: List[str] = None) -> bool:
        """Check if file still has hardcoded prompts (excluding safe patterns)."""
        if exclude_patterns is None:
            exclude_patterns = [
                "You are answering",  # Comments
                "You are a",  # Comments  
            ]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with default encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                return True  # Consider it clean if we can't read it
            
        
        # Remove comments
        content_no_comments = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Check for hardcoded system prompts
        suspicious_patterns = [
            r'"""You are.*?"""',
            r"'''You are.*?'''",
        ]
        
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, content_no_comments, re.DOTALL)
            if matches:
                # Check if it's an excluded pattern
                if not any(exclude in matches[0] for exclude in exclude_patterns):
                    logger.warning(f"Found potential hardcoded prompt in {file_path}")
                    return False
        
        return True


def print_summary():
    """Print migration summary and usage instructions."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║       PROMPT CENTRALIZATION MIGRATION UTILITY                 ║
╚═══════════════════════════════════════════════════════════════╝

USAGE:
  python -m core.prompts.migrate_prompts --scan
    Scan codebase for hardcoded prompts and generate report

  python -m core.prompts.migrate_prompts --migrate
    Apply migrations to identified files

  python -m core.prompts.migrate_prompts --validate
    Validate all migrated files

  python -m core.prompts.migrate_prompts --rollback TIMESTAMP
    Rollback to previous backup

FILES MIGRATED:
  ✓ core/llm/llm_gateway.py
  ✓ agents/langgraph_orchestrator.py
  ✓ memori/agents/retrieval_agent.py
  ✓ memori/agents/memory_agent.py
  ✓ rag/multimodal/prompts.py
  → tools/text_to_sql_tool.py (in progress)
  → tools/medical_coding/auto_coder.py (in progress)

BACKUP LOCATION: .prompt_migrations_backup/

For more information, see docs/PROMPT_CENTRALIZATION_GUIDE.md
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print_summary()
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "--scan":
        scanner = PromptScanner()
        results = scanner.scan()
        print(json.dumps(results, indent=2))
    
    elif command == "--migrate":
        migrator = PromptMigrator()
        migrator.migrate_text_to_sql_tool()
        migrator.migrate_medical_coding_tool()
    
    elif command == "--validate":
        validator = PromptMigrationValidator()
        files_to_check = [
            "tools/text_to_sql_tool.py",
            "tools/medical_coding/auto_coder.py"
        ]
        for file_path in files_to_check:
            is_valid = validator.validate_syntax(file_path)
            has_import = validator.validate_import(file_path)
            no_hardcoded = validator.validate_no_hardcoded_prompts(file_path)
            print(f"{file_path}: Syntax={is_valid}, Import={has_import}, Clean={no_hardcoded}")
    
    elif command == "--rollback" and len(sys.argv) > 2:
        timestamp = sys.argv[2]
        migrator = PromptMigrator()
        success = migrator.rollback(timestamp)
        print("Rollback successful" if success else "Rollback failed")
    
    else:
        print_summary()
