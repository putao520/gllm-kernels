#!/bin/bash
# Download precompiled Metal kernels from GitHub Actions
#
# Usage: ./scripts/download_metal_kernels.sh [--run-id ID]
#
# Prerequisites:
# - GitHub CLI (gh) installed and authenticated
#
# This script downloads the metal-kernels-embedded artifact from the latest
# successful workflow run and places the metallib files in src/metal_kernels/kernels/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_ROOT/src/metal_kernels/kernels"
TEMP_DIR=$(mktemp -d)
WORKFLOW_NAME="compile-kernels.yml"
ARTIFACT_NAME="metal-kernels-embedded"

# Parse arguments
RUN_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--run-id ID]"
            echo ""
            echo "Download precompiled Metal kernels from GitHub Actions."
            echo ""
            echo "Options:"
            echo "  --run-id ID   Specify workflow run ID (default: latest successful)"
            echo ""
            echo "Prerequisites:"
            echo "  - GitHub CLI (gh) installed and authenticated"
            echo "  - Run 'gh auth login' if not authenticated"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for gh CLI
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) not found."
    echo "Install: https://cli.github.com/"
    echo ""
    echo "macOS:   brew install gh"
    echo "Linux:   sudo apt install gh"
    echo "Windows: winget install GitHub.cli"
    exit 1
fi

# Check gh auth status
if ! gh auth status &> /dev/null; then
    echo "Error: GitHub CLI not authenticated."
    echo "Run: gh auth login"
    exit 1
fi

echo "=== Download Metal Kernels from GitHub Actions ==="
echo "Target directory: $TARGET_DIR"
echo ""

# Get the run ID if not specified
if [[ -z "$RUN_ID" ]]; then
    echo "Finding latest successful workflow run..."
    RUN_ID=$(gh run list \
        --workflow="$WORKFLOW_NAME" \
        --status=success \
        --limit=1 \
        --json databaseId \
        --jq '.[0].databaseId')

    if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
        echo "Error: No successful workflow runs found."
        echo ""
        echo "You can trigger a workflow run manually:"
        echo "  gh workflow run $WORKFLOW_NAME --field backend=metal"
        echo ""
        echo "Then wait for it to complete and run this script again."
        exit 1
    fi
fi

echo "Using workflow run ID: $RUN_ID"
echo ""

# Get run info
RUN_INFO=$(gh run view "$RUN_ID" --json status,conclusion,createdAt,headSha)
RUN_STATUS=$(echo "$RUN_INFO" | jq -r '.status')
RUN_CONCLUSION=$(echo "$RUN_INFO" | jq -r '.conclusion')
RUN_DATE=$(echo "$RUN_INFO" | jq -r '.createdAt')
RUN_SHA=$(echo "$RUN_INFO" | jq -r '.headSha')

echo "Run status: $RUN_STATUS ($RUN_CONCLUSION)"
echo "Run date: $RUN_DATE"
echo "Commit: ${RUN_SHA:0:7}"
echo ""

if [[ "$RUN_CONCLUSION" != "success" ]]; then
    echo "Warning: Run did not complete successfully (conclusion: $RUN_CONCLUSION)"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Download artifact
echo "Downloading artifact: $ARTIFACT_NAME..."
cd "$TEMP_DIR"

if ! gh run download "$RUN_ID" --name "$ARTIFACT_NAME" --dir "$TEMP_DIR"; then
    echo "Error: Failed to download artifact."
    echo ""
    echo "This could mean:"
    echo "  1. The artifact has expired (retention: 90 days)"
    echo "  2. The Metal compilation job failed or was skipped"
    echo "  3. Network issues"
    echo ""
    echo "Try triggering a new workflow run:"
    echo "  gh workflow run $WORKFLOW_NAME --field backend=metal"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# List downloaded files
echo ""
echo "Downloaded files:"
find "$TEMP_DIR" -type f -name "*.metallib" -exec ls -lh {} \;

# Check if metallib files exist
METALLIB_COUNT=$(find "$TEMP_DIR" -type f -name "*.metallib" | wc -l)
if [[ "$METALLIB_COUNT" -eq 0 ]]; then
    echo "Error: No metallib files found in artifact."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copy to target directory
echo ""
echo "Copying metallib files to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

# Handle both flat and nested directory structures
find "$TEMP_DIR" -type f -name "*.metallib" -exec cp {} "$TARGET_DIR/" \;

# Verify
echo ""
echo "=== Verification ==="
echo "Installed metallib files:"
ls -lh "$TARGET_DIR"/*.metallib 2>/dev/null || echo "(no metallib files)"

# Check file sizes
echo ""
echo "File sizes:"
for f in "$TARGET_DIR"/*.metallib; do
    if [[ -f "$f" ]]; then
        size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
        if [[ "$size" -eq 0 ]]; then
            echo "  WARNING: $f is empty!"
        else
            echo "  $(basename "$f"): $size bytes"
        fi
    fi
done

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Done ==="
echo ""
echo "Metal kernels are now ready for embedding."
echo "The next 'cargo build' will include these precompiled kernels."
echo ""
echo "To verify embedding in Rust code:"
echo '  const DATA: &[u8] = include_bytes!("kernels/flash_attention.metallib");'
echo '  assert!(!DATA.is_empty(), "metallib not embedded");'
