import json
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_DIRECTORY = Path("plots")


@dataclass
class TagVersions:
    old: Optional[Dict[str, List[str]]]
    current: Optional[Dict[str, List[str]]]
    new: Optional[Dict[str, List[str]]]

    @classmethod
    def from_files(
        cls,
        old_path: Optional[Path] = None,
        current_path: Optional[Path] = None,
        new_path: Optional[Path] = None,
    ) -> "TagVersions":
        try:
            return cls(
                old=json.loads(old_path.read_text()) if old_path else None,
                current=json.loads(current_path.read_text()) if current_path else None,
                new=json.loads(new_path.read_text()) if new_path else None,
            )
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load tag data: {e}")

    def filter_by_step(self, step: Optional[int]) -> "TagVersions":
        if step is None:
            return TagVersions(
                old=copy(self.old), current=copy(self.current), new=copy(self.new)
            )
        return TagVersions(
            old=_only_keep_step(self.old, step),
            current=_only_keep_step(self.current, step),
            new=_only_keep_step(self.new, step),
        )

    def get_tag_data_dict(self) -> dict:
        """Get tag data for all versions."""
        return {
            name: data
            for name, data in {
                "Original": self.old,
                "Current": self.current,
                "New": self.new,
            }.items()
            if data is not None
        }


def _only_keep_step(note_id_to_tags: Optional[dict], usmle_step: int) -> Optional[dict]:
    if note_id_to_tags is None:
        return None
    return {
        note_id: [tag for tag in tags if f"step{usmle_step}" in tag.lower()]
        for note_id, tags in note_id_to_tags.items()
    }


def print_note_tag_stats(note_id_to_tags: dict):
    # Convert to DataFrame of note_id, tag pairs
    note_id_to_tags_df = pd.DataFrame(
        [(note_id, tag) for note_id, tags in note_id_to_tags.items() for tag in tags],
        columns=["note_id", "tag"],
    )

    # Get number of notes with tags
    num_notes_with_tags = len(note_id_to_tags)

    # Get number of unique tags
    num_unique_tags = len(note_id_to_tags_df["tag"].unique())

    # Total number of tags
    total_tags = len([tag for tags in note_id_to_tags.values() for tag in tags])

    # Print results
    print(f"{num_notes_with_tags} notes have tags")
    print(f"{num_unique_tags} unique tags are used")
    print(f"{total_tags} tags in total")
    print()

    # Get average number of tags per note
    avg_tags_per_note = note_id_to_tags_df.groupby("note_id").size().mean()

    # Get median number of tags per note
    median_tags_per_note = note_id_to_tags_df.groupby("note_id").size().median()

    # Get max number of tags per note
    max_tags_per_note = note_id_to_tags_df.groupby("note_id").size().max()

    # Print results
    print(
        f"Average number of tags per note (with at least one tag): {avg_tags_per_note:.2f}"
    )
    print(
        f"Median number of tags per note (with at least one tag): {median_tags_per_note}"
    )
    print(f"Max number of tags per note: {max_tags_per_note}")
    print()

    # Get average number of notes per tag
    avg_notes_per_tag = note_id_to_tags_df.groupby("tag").size().mean()

    # Get median number of notes per tag
    median_notes_per_tag = note_id_to_tags_df.groupby("tag").size().median()

    # Calculate notes per tag
    notes_per_tag = note_id_to_tags_df.groupby("tag").size()

    # Get max number of notes per tag
    max_notes_per_tag = notes_per_tag.max()

    # Print results
    print(f"Average number of notes per tag: {avg_notes_per_tag:.2f}")
    print(f"Median number of notes per tag: {median_notes_per_tag}")
    print(f"Max number of notes per tag: {max_notes_per_tag}")


def create_comparison_box_plots(
    tag_versions: TagVersions,
    usmle_step: Optional[int],
) -> None:
    """Create box plots comparing tag distributions across different versions."""
    # Create DataFrames for each version
    tags_by_version = tag_versions.get_tag_data_dict()
    dfs = [
        _create_tags_dataframe(tags, version)
        for version, tags in tags_by_version.items()
        if tags
    ]

    # Combine data and create plot
    combined_data = pd.concat(dfs)
    _create_box_plot(
        combined_data, usmle_step, version_names=list(tags_by_version.keys())
    )


def _create_tags_dataframe(note_id_to_tags: dict, version: str) -> pd.DataFrame:
    """Create and label a DataFrame for a specific tag version."""
    df = pd.DataFrame(
        [(note_id, tag) for note_id, tags in note_id_to_tags.items() for tag in tags],
        columns=["note_id", "tag"],
    )
    notes_per_tag = df.groupby("tag").size().reset_index(name="count")
    notes_per_tag["version"] = version
    return notes_per_tag


def _create_box_plot(
    combined_data: pd.DataFrame, usmle_step: Optional[int], version_names: List[str]
) -> None:
    """Create and save a box plot comparing tag distributions."""
    # Adjust figure size based on number of versions
    if len(version_names) == 1:
        # Narrower figure for single version
        plt.figure(figsize=(6, 6))  # Square aspect ratio for single version
    else:
        # Standard wider figure for multiple versions
        plt.figure(figsize=(12, 6))

    # Adjust boxplot width and appearance
    if len(version_names) == 1:
        # For single version, control width with width parameter
        ax = sns.boxplot(
            data=combined_data,
            x="version",
            y="count",
            width=0.3,  # Narrower box for single version
            color="#1f77b4",  # Consistent color with histogram
        )

        # Adjust x-axis limits to prevent box from being too wide
        x_min, x_max = ax.get_xlim()
        padding = (x_max - x_min) * 0.4  # 40% padding on each side
        ax.set_xlim(x_min - padding, x_max + padding)

        # Adjusted title for single version
        title = f"Distribution of Notes per Tag - {version_names[0]}"
    else:
        # Multiple versions - use default width
        sns.boxplot(data=combined_data, x="version", y="count")

        # Title with comparison
        title = "Distribution of Notes per Tag - " + " vs ".join(version_names)

    if usmle_step:
        title += f" - USMLE Step {usmle_step}"

    plt.title(title)
    plt.xlabel("Version")
    plt.ylabel("Number of Notes")

    # Make sure grid is only on y-axis and subtle
    plt.grid(axis="y", alpha=0.3)

    # Make plot more compact by adjusting layout
    plt.tight_layout()

    filename = (
        f"notes_per_tag_comparison{'_step' + str(usmle_step) if usmle_step else ''}.png"
    )
    plt.savefig(PLOT_DIRECTORY / filename)
    plt.close()


def create_tags_per_note_histogram(
    tag_versions: TagVersions,
    usmle_step: Optional[int],
) -> None:
    """Create histograms comparing tags per note distributions across versions."""
    # Get number of tags per note for each version
    tag_amounts_by_version = {
        version: [len(tags) for tags in note_id_to_tags.values()]
        for version, note_id_to_tags in tag_versions.get_tag_data_dict().items()
    }
    plt.figure(figsize=(12, 6))
    max_tags = max(max(data) for data in tag_amounts_by_version.values() if data)
    bins = np.arange(0, max_tags + 2)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # Blue, Green, Orange

    # Determine number of versions and adjust bar width and positioning
    num_versions = len(tag_amounts_by_version)

    if num_versions == 1:
        # For single version, use wider bars centered on tick marks
        bar_width = 0.6
        for idx, (version, data) in enumerate(tag_amounts_by_version.items()):
            counts, edges = np.histogram(data, bins=bins)
            # Center the bars directly on tick marks
            plt.bar(
                edges[:-1],  # Center on actual bin positions
                counts,
                width=bar_width,
                label=version,
                color=colors[idx],
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )
    else:
        # For multiple versions, create grouped bars
        bar_width = 0.8 / num_versions  # Adjust width based on number of versions

        for idx, (version, data) in enumerate(tag_amounts_by_version.items()):
            counts, edges = np.histogram(data, bins=bins)
            # Calculate position to center group of bars around tick marks
            offset = bar_width * (idx - (num_versions - 1) / 2)
            centers = edges[:-1] + offset

            plt.bar(
                centers,
                counts,
                width=bar_width,
                label=version,
                color=colors[idx],
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )

    # Adjust title based on number of versions
    versions = list(tag_versions.get_tag_data_dict().keys())
    if len(versions) == 1:
        title = f"Distribution of Tags per Note - {versions[0]}"
    else:
        title = "Distribution of Tags per Note - " + " vs ".join(versions)

    if usmle_step:
        title += f" - USMLE Step {usmle_step}"

    plt.title(title)
    plt.xlabel("Number of Tags")
    plt.ylabel("Number of Notes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(bins[:-1])

    filename = (
        f"tags_per_note_comparison{'_step' + str(usmle_step) if usmle_step else ''}.png"
    )
    plt.savefig(PLOT_DIRECTORY / filename)
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--old",
        type=Path,
        help="Path to the old note ID to tags JSON file",
    )
    parser.add_argument(
        "-c",
        "--current",
        type=Path,
        help="Path to the current note ID to tags JSON file",
    )
    parser.add_argument(
        "-n",
        "--new",
        type=Path,
        help="Path to the new note ID to tags JSON file",
    )
    args = parser.parse_args()

    tags = TagVersions.from_files(
        old_path=args.old,
        current_path=args.current,
        new_path=args.new,
    )

    for step in [None, 1, 2]:
        filtered_tag_versions = tags.filter_by_step(step)
        create_comparison_box_plots(filtered_tag_versions, step)
        create_tags_per_note_histogram(filtered_tag_versions, step)

    for version, tags in tags.get_tag_data_dict().items():
        if not tags:
            continue

        print(f"\n{version} tags:")
        print_note_tag_stats(tags)


if __name__ == "__main__":
    main()
