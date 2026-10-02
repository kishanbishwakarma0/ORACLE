
from src.active_learning.label_store import LabelStore


def main():

    store = LabelStore()

    # Add initial annotations
    initial_annotations = {
        0: 2,
        1: 5,
        2: 1,
    }

    store.add(initial_annotations)

    assert store.get_annotation_count() == 3
    assert store.get_label(1) == 5
    assert store.is_annotated(2)
    assert not store.is_annotated(10)

    # Add new annotations
    store.add({
        3: 7,
        4: 9,
    })

    assert store.get_annotation_count() == 5

    # Verify duplicate protection
    try:
        store.add({1: 3})
    except ValueError:
        print("Duplicate label protection passed!")
    else:
        raise AssertionError(
            "Duplicate label was not blocked."
        )

    # Verify hidden labels cannot be accessed
    try:
        store.get_label(10)
    except KeyError:
        print("Unannotated label access protection passed!")
    else:
        raise AssertionError(
            "Unannotated label access was not blocked."
        )

    print("\n--- Label Store Test ---")
    print("Stored annotations:", store.get_all_labels())
    print("Annotation count:", store.get_annotation_count())
    print("\nAll Label Store tests passed!")


if __name__ == "__main__":
    main()