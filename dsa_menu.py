#!/usr/bin/env python3
"""
DSA Menu
--------
Interactive CLI to view small data structures & algorithms code snippets
and run tiny demos. Perfect for quick revision.

Run:
  python dsa_menu.py
"""

from collections import deque
from typing import List, Dict, Optional, Tuple


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_snippet(title: str, code: str) -> None:
    print_header(f"Snippet: {title}")
    print(code)
    print("-" * 72)


# 1) Binary Search (Iterative)
def demo_binary_search() -> None:
    code = """
def binary_search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""".strip("\n")

    show_snippet("Binary Search (Iterative)", code)

    def binary_search(nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    nums = [1, 3, 5, 7, 9, 11]
    for t in [7, 2, 11]:
        print(f"Search {t} in {nums} -> index {binary_search(nums, t)}")


# 2) Merge Sort
def demo_merge_sort() -> None:
    code = """
def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    i = j = 0
    result: List[int] = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""".strip("\n")

    show_snippet("Merge Sort", code)

    def merge_sort(arr: List[int]) -> List[int]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        return merge(left, right)

    def merge(left: List[int], right: List[int]) -> List[int]:
        i = j = 0
        result: List[int] = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i]); i += 1
            else:
                result.append(right[j]); j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    arr = [5, 2, 4, 6, 1, 3]
    print(f"Before: {arr}")
    print(f"After:  {merge_sort(arr)}")


# 3) Singly Linked List (Insert Head, Print)
def demo_singly_linked_list() -> None:
    code = """
class Node:
    def __init__(self, val: int, next: Optional['Node']=None):
        self.val = val
        self.next = next

class SinglyLinkedList:
    def __init__(self):
        self.head: Optional[Node] = None

    def insert_head(self, val: int) -> None:
        self.head = Node(val, self.head)

    def to_list(self) -> List[int]:
        out: List[int] = []
        curr = self.head
        while curr:
            out.append(curr.val)
            curr = curr.next
        return out
""".strip("\n")

    show_snippet("Singly Linked List (insert head, print)", code)

    class Node:
        def __init__(self, val: int, next: Optional['Node'] = None):
            self.val = val
            self.next = next

    class SinglyLinkedList:
        def __init__(self):
            self.head: Optional[Node] = None

        def insert_head(self, val: int) -> None:
            self.head = Node(val, self.head)

        def to_list(self) -> List[int]:
            out: List[int] = []
            curr = self.head
            while curr:
                out.append(curr.val)
                curr = curr.next
            return out

    ll = SinglyLinkedList()
    for v in [3, 2, 1]:
        ll.insert_head(v)
    print(f"LinkedList -> {ll.to_list()} (inserted 1,2,3 at head)")


# 4) Stack using Python list
def demo_stack() -> None:
    code = """
class Stack:
    def __init__(self) -> None:
        self._data: List[int] = []

    def push(self, x: int) -> None:
        self._data.append(x)

    def pop(self) -> int:
        return self._data.pop()

    def peek(self) -> int:
        return self._data[-1]

    def is_empty(self) -> bool:
        return not self._data
""".strip("\n")

    show_snippet("Stack (list-backed)", code)

    class Stack:
        def __init__(self) -> None:
            self._data: List[int] = []

        def push(self, x: int) -> None:
            self._data.append(x)

        def pop(self) -> int:
            return self._data.pop()

        def peek(self) -> int:
            return self._data[-1]

        def is_empty(self) -> bool:
            return not self._data

    st = Stack()
    for x in [1, 2, 3]:
        st.push(x)
    print("Popped:", st.pop())
    print("Peek:  ", st.peek())
    print("Empty?:", st.is_empty())


# 5) Queue using deque
def demo_queue() -> None:
    code = """
from collections import deque

class Queue:
    def __init__(self) -> None:
        self._dq: deque[int] = deque()

    def enqueue(self, x: int) -> None:
        self._dq.append(x)

    def dequeue(self) -> int:
        return self._dq.popleft()

    def is_empty(self) -> bool:
        return not self._dq
""".strip("\n")

    show_snippet("Queue (deque-backed)", code)

    class Queue:
        def __init__(self) -> None:
            self._dq: deque[int] = deque()

        def enqueue(self, x: int) -> None:
            self._dq.append(x)

        def dequeue(self) -> int:
            return self._dq.popleft()

        def is_empty(self) -> bool:
            return not self._dq

    q = Queue()
    for x in [10, 20, 30]:
        q.enqueue(x)
    print("Dequeued:", q.dequeue())
    print("Empty?:  ", q.is_empty())


# 6) Graph BFS (Adjacency List)
def demo_bfs() -> None:
    code = """
from collections import deque

def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    visited = set([start])
    order: List[int] = []
    dq = deque([start])
    while dq:
        node = dq.popleft()
        order.append(node)
        for nei in graph.get(node, []):
            if nei not in visited:
                visited.add(nei)
                dq.append(nei)
    return order
""".strip("\n")

    show_snippet("Graph BFS", code)

    def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
        visited = set([start])
        order: List[int] = []
        dq = deque([start])
        while dq:
            node = dq.popleft()
            order.append(node)
            for nei in graph.get(node, []):
                if nei not in visited:
                    visited.add(nei)
                    dq.append(nei)
        return order

    graph = {
        0: [1, 2],
        1: [2],
        2: [0, 3],
        3: [3],
    }
    print("BFS from 2 ->", bfs(graph, 2))


# 7) DFS (Recursive)
def demo_dfs() -> None:
    code = """
def dfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    visited = set()
    order: List[int] = []

    def rec(node: int) -> None:
        visited.add(node)
        order.append(node)
        for nei in graph.get(node, []):
            if nei not in visited:
                rec(nei)

    rec(start)
    return order
""".strip("\n")

    show_snippet("Graph DFS (recursive)", code)

    def dfs(graph: Dict[int, List[int]], start: int) -> List[int]:
        visited = set()
        order: List[int] = []

        def rec(node: int) -> None:
            visited.add(node)
            order.append(node)
            for nei in graph.get(node, []):
                if nei not in visited:
                    rec(nei)

        rec(start)
        return order

    graph = {
        0: [1, 2],
        1: [2],
        2: [0, 3],
        3: [3],
    }
    print("DFS from 2 ->", dfs(graph, 2))


# 8) Binary Tree Traversals
def demo_tree_traversals() -> None:
    code = """
class TreeNode:
    def __init__(self, val: int, left: Optional['TreeNode']=None, right: Optional['TreeNode']=None):
        self.val, self.left, self.right = val, left, right

def inorder(root: Optional[TreeNode]) -> List[int]:
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

def preorder(root: Optional[TreeNode]) -> List[int]:
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []

def postorder(root: Optional[TreeNode]) -> List[int]:
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []
""".strip("\n")

    show_snippet("Binary Tree Traversals", code)

    class TreeNode:
        def __init__(self, val: int, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
            self.val, self.left, self.right = val, left, right

    def inorder(root: Optional[TreeNode]) -> List[int]:
        return inorder(root.left) + [root.val] + inorder(root.right) if root else []

    def preorder(root: Optional[TreeNode]) -> List[int]:
        return [root.val] + preorder(root.left) + preorder(root.right) if root else []

    def postorder(root: Optional[TreeNode]) -> List[int]:
        return postorder(root.left) + postorder(root.right) + [root.val] if root else []

    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    print("Inorder:  ", inorder(root))
    print("Preorder: ", preorder(root))
    print("Postorder:", postorder(root))


# 9) Dijkstra's Shortest Path (using adjacency list with weights)
def demo_dijkstra() -> None:
    code = """
import heapq

def dijkstra(graph: Dict[int, List[Tuple[int, int]]], src: int) -> Dict[int, int]:
    dist: Dict[int, int] = {node: float('inf') for node in graph}
    dist[src] = 0
    pq: List[Tuple[int, int]] = [(0, src)]
    while pq:
        d, node = heapq.heappop(pq)
        if d != dist[node]:
            continue
        for nei, w in graph.get(node, []):
            nd = d + w
            if nd < dist[nei]:
                dist[nei] = nd
                heapq.heappush(pq, (nd, nei))
    return dist
""".strip("\n")

    show_snippet("Dijkstra's Algorithm", code)

    import heapq

    def dijkstra(graph: Dict[int, List[Tuple[int, int]]], src: int) -> Dict[int, int]:
        dist: Dict[int, int] = {node: float('inf') for node in graph}
        dist[src] = 0
        pq: List[Tuple[int, int]] = [(0, src)]
        while pq:
            d, node = heapq.heappop(pq)
            if d != dist[node]:
                continue
            for nei, w in graph.get(node, []):
                nd = d + w
                if nd < dist[nei]:
                    dist[nei] = nd
                    heapq.heappush(pq, (nd, nei))
        return dist

    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: [],
    }
    print("Distances from 0:", dijkstra(graph, 0))


DSA_MENU_ITEMS = [
    ("Binary Search (Iterative)", demo_binary_search),
    ("Merge Sort", demo_merge_sort),
    ("Singly Linked List", demo_singly_linked_list),
    ("Stack (list)", demo_stack),
    ("Queue (deque)", demo_queue),
    ("Graph BFS", demo_bfs),
    ("Graph DFS (recursive)", demo_dfs),
    ("Binary Tree Traversals", demo_tree_traversals),
    ("Dijkstra's Algorithm", demo_dijkstra),
]


def print_dsa_menu() -> None:
    print("\nChoose a DSA snippet to view & demo:")
    for i, (label, _) in enumerate(DSA_MENU_ITEMS, start=1):
        print(f"  {i}. {label}")
    print("  0. Back")


def run_dsa_snippets_menu() -> None:
    while True:
        print_dsa_menu()
        try:
            choice = int(input("Enter choice: ").strip())
        except ValueError:
            print("Please enter a valid number.")
            continue

        if choice == 0:
            break
        if 1 <= choice <= len(DSA_MENU_ITEMS):
            label, action = DSA_MENU_ITEMS[choice - 1]
            try:
                action()
            except Exception as exc:
                print(f"Error while running '{label}': {exc}")
        else:
            print("Invalid choice. Try again.")


def print_top_menu() -> None:
    print("\nSelect an option:")
    print("  1. Code-snippets")
    print("  2. local-code-snippets")
    print("  3. Static code analysis")
    print("  0. exit")


def run_local_code_snippets() -> None:
    print_header("Local Code Snippets")
    print("Placeholder: point this to your local snippets folder later.")
    print("Tip: You can integrate file loading here to display your own code.")


def run_static_code_analysis() -> None:
    print_header("Static Code Analysis")
    try:
        from static_analysis import run_static_code_analysis as _run
        path = input("Path to analyze (default=.): ").strip() or "."
        _run(path)
    except Exception as exc:
        print(f"Error running analysis: {exc}")


def main() -> None:
    print_header("DSA Learning Hub")
    while True:
        print_top_menu()
        try:
            choice = int(input("Enter choice: ").strip())
        except ValueError:
            print("Please enter a valid number.")
            continue

        if choice == 0:
            print("Goodbye!")
            break
        elif choice == 1:
            print_header("DSA Snippets & Demos")
            run_dsa_snippets_menu()
        elif choice == 2:
            run_local_code_snippets()
        elif choice == 3:
            run_static_code_analysis()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()


