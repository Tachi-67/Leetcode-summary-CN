## Array (Python: [])
1. 要使用的数据具有*单调结构*（e.g.需要直到idx的最大值，直到idx的乘积）时考虑：
      - Precomputation, 先过一遍数组，获得相关信息。 
      - **双指针**
2. 在处理*子数组，子区间*问题时，考虑使用：
      - **滑动窗口**：可以聚集连续子区间内的信息。
      - **Two pointers**: 
      - kadane's algorithm -> dp.
3. 关于**双指针**：
      - two sum, three sum 问题
      - 需要比较两个“点”，且两个点的关系随算法的进行改变
      - 如果正向遍历遇到困难，总是可以考虑反向遍历。
      - 双指针带来的是**两个数据点**，如果要考虑对数据点进行操作的话，注意此时可能有两种选择 -> 可能需要考虑递归搜索。
4. 关于**滑动区间**：
      - **'subarray'**
      - 滑动区间算法成立的必要条件是：符合条件的子区间可以通过延续至后续元素的方式变得更长；不符合的子空间，其后续延长也不能合法。

        - 简单地说，子区间的合法性推导必须满足：**YES -> (YES or NO); NO -> NO**
        - **如果问题满足NO -> YES的情况，则不能使用滑动区间。**
        - 例子1：子区间极差 <= 2
          - [6, 4, 5] 满足 -> [6, 4, 5, ?]可能满足，也可能不满足。符合条件；
        - 例子2：子区间unique element数量 <= 2
          - [2, 1, 1] 不满足（仅有2是unique的） -> [2, 1, 1, 3] 满足（2，3 unique），不符合条件。 
      - 滑动区间的重点在于找到有效的办法来收集&更新区间内的信息
        - 一个子问题模板：记录sliding interval中的元素最小值
          - 使用heap来追踪最小值
          - 在查询之前，pop直到堆顶元素idx在区间之内
      - 一个滑动空间的implementation模板：
        - Keep moving r by loop;
        - Move l until [l...r] is valid;
        - Update answer.
5. 当数据位置不重要的时候，总是可以考虑**排序**。
6. 关于**二分查找**：
      - 二分查找某个数值：
      ```
      # target在闭区间[left, right]内
      while left <= right:
        if mid satisfies condition:
          return mid
        # 以下：target严格大于小于arr[mid]，则一定不在以mid为界的区间中
        if arr[mid] < target:
          left = mid + 1
        if arr[mid] > target:
          right = mid - 1
      return -1 #没有找到
      ```
      - 二分查找第一个值为target的元素，同理可以写二分查找最后一个值为target的元素
      ```
      while left <= right:
        if a[mid] == target:
          if mid == 0 or a[mid - 1] != target:
            return mid
          else:
            right = mid - 1
        if a[mid] < target:
          left = mid + 1
        if a[mid] > target:
          right = mid - 1
      return -1
      ```
      - 二分查找第一个>=target的元素，同理可写最后一个<=target的元素
      ```
      while left <= right:
        if a[mid] >= target:
          if mid == 0 or a[mid - 1] < target:
            return mid
          else:
            right = mid - 1
        else: # a[mid] < target
          left = mid + 1
      return -1
      ```
      - **注意：** `1e10`这样的数字是一个浮点数，如果把`r`设置为了`1e10`，则`mid = (l + r) // 2`会得到一个浮点数。如果最后需要返回的mid是一个整数，这里需要注意进行转换。
        
7. 往一个array上面一次添加多个元素： `arr + arr_to_add` e.g. `[1, 2] + [3, 4] = [1, 2, 3, 4]`
8. 一个元素重复，长为`len_arr`的array: `[element] * len_arr` e.g. `[2] * 3 == [2, 2, 2]`
9. 关于**'circular array'** (循环数组)的情况，一般的好做法是给原数组append一个一样的数组，这样就解决了循环问题。
10. 从array里面移除一个元素，通常时间复杂度是O(n), 但是也有一种方法，即：
      - 将要删除的元素和最后一个元素交换
      - pop

     这样子会打乱array的顺序，但是如果我们额外有一个`dict`来记录数据的index（当数字unique），则没有问题
   
## String
1. https://www.techinterviewhandbook.org/algorithms/string/
2. String和Array的trick有共通之处，尤其是**滑动区间**和**Two pointers**
3. 注意关注anagram和palindrome的定义。
4. char -> int: ord(char) - ord('a')，这可以导出string -> 质因数分解、bitmask（仅适用于string of unique cahracters）
5. int -> char: chr(int)
   
## Hash Tables (Python: {})
1. https://www.techinterviewhandbook.org/algorithms/hash-table/
2. 移除{}中的某个元素：`del hash_map[index]`
   - **注意！** 该操作需要谨慎进行，特别是在循环中作为终止条件的时候，如果提早del掉可能导致`TypeError`
          - 处理办法是再进入循环前提前记录这个值。
4. `dict.get(key, alternative_value)`
5. `dict.keys()`; `dict.values()`;`dict.items() -> dictitems, use list to convert`
   
## Recursion
1. https://www.techinterviewhandbook.org/algorithms/recursion/
2. 关于**暴力搜索路径查找**：
   1. Leetcode 236, 39
   2. 注意，如果说要把path加入递归函数的参数，因为path（一般而言是一个list）是可变的，所以path会被所有递归函数共享
   3. 把path作为递归参数传入时，在当前节点，如果节点非空则将节点加入path，搜索完毕后把当前节点去除。
      ```
      def search(cur, target, path):
        if cur is None:
          # search fails here
        path.append(cur) # add cur to current path
        if cur is target:
          # search succeeds here
          # 如果要在这里记录path，请使用path.copy()以避免path被其他递归分支修改。
        # look at cur's neighbors and continue search
        # do something...
        path.pop() # quitting search from cur, pop cur from path
        # if all search fails, do something here
      ```
    4. 如果不想使用共享path，也可以：
        ```
        def search(cur, target, path):
          if cur is None:
            # search fails here
          path = path + [cur] # 这样的写法创建了新的path，与参数path无关
          if cur is target:
            # search succeeds here
            # 可以直接记录path
          # look at cur's neighbors and continue search
          # do something...
          # if all search fails, do something here
        ```
        这样做的缺点是会增加空间使用量，在较大数据集时可能MLE
## Sorting and Searching 
1. 见Array - 二分查找
2. 实现Merge Sort和Quick Sort，他们都是基于divide现有array再进行递归执行的算法，区别在于: Merge Sort基于array的位置中点进行divide；QSort则利用array元素与选取pivot的大小比较进行divide.

    **TRICK:** 考虑长度为2的数组，有利于构造edge case

    merge sort对数组基于index进行分割，分割为等大的两个subarray。需要进行O(n)的**合并**。
    ```
    # merge sort: divide based on position (mid)
    import math
    MAX_NUM = 101
    def merge_sort(arr):
      # base case
      if len(arr) <= 1:
          return arr
      mid = len(arr) // 2
      # 注意，这里不能使用(len(arr) - 1) // 2，考虑长度为2的数组，这样分割不会停止。
      # 这区分于二分查找的情况，因为在二分查找里，我们会直接l = mid - 1 or r = mid + 1
      #mid = math.ceil((len(arr)-1)/2)
      arr_l = arr[:mid]
      arr_r = arr[mid:]
      sorted_l = merge_sort(arr_l)
      sorted_r = merge_sort(arr_r)
      sorted_l.append(MAX_NUM)
      sorted_r.append(MAX_NUM)
      pointer_l = 0
      pointer_r = 0
      ret_arr = []
      while len(ret_arr) < len(arr):
          if sorted_l[pointer_l] <= sorted_r[pointer_r]:
              ret_arr.append(sorted_l[pointer_l])
              pointer_l += 1
          else:
              ret_arr.append(sorted_r[pointer_r])
              pointer_r += 1
      return ret_arr
    ```

    quick sort对数组进行基于pivot数值的分割。
    进行分割时需要注意的是，要把原数组分成3份(<, =, >)，否则会导致递归爆栈（考虑list = [2, 1]）
    ```
    # quick sort: divide based on pivot value
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr)//2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    ```
  3. 关于'第K大'的问题：可以考虑使用`heapq`, 关于其的使用方法，见下面。
## Matrix
1. https://www.techinterviewhandbook.org/algorithms/matrix/
2. 创建一个m * n的矩阵：
    ```
    [[0 for _ in range(n)] for _ in range(m)]
    ```
3. 求矩阵的转置:
    ```
    for i in range(n):
      for j in range(i, n):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    ```
    如果不是方阵的话，则需要另外创建一个matrix来进行赋值。
## Linked list
1. https://www.techinterviewhandbook.org/algorithms/linked-list/
2. 实用提醒：
  * 设置dummy node: 单向链表设置一个tail，双向链表设置head和tail。如果最后要返回链表，记得移除他们。
  * 双指针在链表中应用很多，见网页。
3. 定义linked list(单向链表，如果是双向的话还要加上self.prev = prev)：
   ```
   class ListNode:
    def __init__(self, val = 0, next = None):
      self.val = val
      self.next = next
   ```
4. 遍历linked list:
    ```
    pointer = head
    while pointer:
      # do something
      pointer = pointer.next
    ```
5. in-place reverse:
    ```
    prev = None
    cur = head
    while cur:
      next = cur.next
      cur.next = prev
      prev = cur
      cur = next
    return prev
    ```
6. find middle point， 如果是偶数长度的链表则返回两个中间点中的第二个:
    ```
    pointer1 = head
    pointer2 = head
    while pointer2 and pointer2.next:
      pointer1 = pointer1.next
      pointer2 = pointer2.next.next
    return pointer1
    ```
    如果想要找的是偶数链表中两个中间点中的第一个：
    ```
    pointer1 = head
    pointer2 = head
    while pointer2.next and pointer2.next.next:
      pointer1 = pointer1.next
      pointer2 = pointer2.next.next
    return pointer1
    ```
7. Merge two lists (https://leetcode.com/problems/merge-two-sorted-lists/):
    ```
    pointer1 = list1
    pointer2 = list2
    while pointer1 and pointer2:
          if pointer1.val <= pointer2.val:
              head.next = pointer1
              head = head.next
              pointer1 = pointer1.next
          else:
              head.next = pointer2
              head = head.next
              pointer2 = pointer2.next
      if pointer1:
          head.next = pointer1
      if pointer2:
          head.next = pointer2
    ```
## Queue (Python: queue)
1. 初始化一个queue，以及基本操作
    ```
    q = queue.Queue()
    q.put(x)
    q.qsize()
    q.empty()
    q.get() # 移除并返回一个元素
    ```
2. 使用的更多的：`deque`
   ```
   from collections import deque
   q = deque()
   q.append(1)
   q.append(2)
   print(q.popleft())
   ```
   
## Stack (Python: [])
1. **单调栈**的技巧：
    * 在栈中维持单调的顺序，常常用于解决**在数组中对于每个元素找到它的下一个更大/更小的元素的问题**
    * 考虑原数组正向或者反向地入栈。
    * 工作原理（以单减栈 & 正向入栈为例）：
      * 如果栈空或者新元素小于等于栈顶元素，入栈
      * 如果新元素大于栈顶元素，pop直到栈空或者栈顶大于等于新元素。对于每一个弹出的元素，新元素是他的“下一个更大元素”
      * 此时的栈顶元素是新元素往左看第一个比他大的元素（考虑反向入栈，则是新元素往右看第一个比他大的元素）
      * 时间复杂度为O(n)
2. stack可以用于处理_reverse Polish notation_, 但是对于一般的计算，用deque从左到右模拟更合理一些。

   
## Tree
0. 关于edge case和base case
    - **if root is None的情况需要特别注意**
    - 如果cur_node是leaf，那么可能是base case 
1. 提到关于 'top to bottom'或者关于从上到下的垂直顺序的，应该考虑BFS(而不是DFS)
    - BFS一个树：https://leetcode.com/problems/binary-tree-vertical-order-traversal/ 
2. Complete Binary Tree: **All levels are full except the last**, in the last level, the nodes are as far left as possible.
   1. Test Completeness: BFS遍历树的每一层。当遇到一个没有子节点或只有左子节点的节点后，该节点之后遍历到的所有节点都应该是叶子节点，这确保了最后一层的节点集中在左侧并且没有间隔。
3. binary tree的前序、中序、后续遍历。这里的前中后指的是root的位置。
    * 前序遍历：
      ```
      ans = []
      def preorder(cur):
        if cur == None:
          return
        ans.append(cur)
        preorder(cur.left)
        preorder(cur.right)
      ```
    * 中序遍历：二分查找树的中序遍历即输出数列从小到大
      ```
      ans = []
      def inorder(cur):
        if cur == None:
          return
        inorder(cur.left)
        ans.append(cur)
        inorder(cur.right)
      ```
    * 后序遍历：
      ```
      ans = []
      def postorder(cur):
        if cur == None:
          return
        postorder(cur.left)
        postorder(cur.right)
        ans.append(cur)
      ```
    * 关于这三种遍历及他们衍生出的算法，一个比较直观的intuition是**什么时候需要左右子树的信息**，就这而言，一般基于后序遍历的算法会比较常见，因为我们想**获得左右子树的信息之后，进行在本节点的总结和处理，再回到母节点。**
      
4. 在binary tree中搜索指定的节点并返回路径，同样适用于图中的搜索：
      ```
      # 需要注意的是，path在此处被所有递归函数共享，所以需要return copy, 并且如果没有找到的话需要pop当前节点。
      def find_path(cur, target, path):
        if cur == None:
            return None
        path.append(cur)
        if cur == target:
            return path.copy()
        path_l = find_path(cur.left, target, path)
        if path_l:
            return path_l
        path_r = find_path(cur.right, target, path)
        if path_r:
            return path_r
        path.pop()
        return None
      ```
      或者更直观的：
      ```
      def find_path(cur_node, target_node, path):
        if cur_node is None:
            return False, path
        path.append(cur_node)
        if cur_node == target_node:
            return True, path[:] 
        result_l, path_l =  find_path(cur_node.left, target_node, path)
        if result_l:
            return True, path_l
        result_r, path_r = find_path(cur_node.right, target_node, path)
        if result_r:
            return True, path_r
        path.pop()
        return False, path
      ```
5.  在binary tree中，一个重要的观察点是**是否有记录父节点**，如果有的话就可以从节点向上遍历。
6.  关于BST的定义：
      1. 根节点value大于左子树中**所有**value；
      2. 根节点value小于右子树中**所有**value；
      3. 左子树，右子树都满足如上定义
   
7. BST / min heap?
      1. BST根节点大于左子树所有节点，小于右子树所有节点
      2. min heap的根节点为以其为根的树的**最小点**
9.  在**BST**中添加一个节点
      ```
      class TreeNode:
        def __init__(self, value, left: Optional[TreeNode]=None, right: Optional[TreeNode]=None):
          self.value = value
          self.left = left
          self.right = right
      
      # assume we already have a BST with tree root = root: Optional[TreeNode]
      def insert_value(value):
        if root is None:
          root = TreeNode(value)
        else:
          insert_helper(root, value)
      
      def insert_helper(cur_node, value):
        # helper function to insert the value to the correct location
        if cur_node.value < value:
          if cur_node.right is None:
            cur_node.right = TreeNode(value)
          else:
            insert_helper(cur_node.right, value)
        if cur_node.value > value:
          if cur_node.left is None:
            cur_node.left = TreeNode(value)
          else:
            insert_helper(cur_node.left, value)
      ```
10. 对BST进行in-order traversal会得到一个递增序列。为了重建一个平衡的BST，可以从一个递增的序列出发递归构造。 
## Graph
1. 图一般以邻接矩阵和邻接表的形式给出。
2. 给定连接的边，建图： 
   1. 邻接矩阵：直接在对应的矩阵上加值即可。
   2. 邻接表：是一个`dict`: from vertex to list，其中 list存储了当前vertex可达的点。
   ```
    # suppose we have a list of lists: edges, each representing an edge.
    # we build a adjacancy list representing this graph
    edges = [[1,2], [3,4], [1,3]]
    adj = {}
    for edge in edges:
        u, v = edge
        if u not in adj:
            adj[u] = [v]
        else:
            adj[u].append(v)
    if v not in adj: # make sure ALL vertexes are in the adj list
        adj[v] = {}

    # Now we do a DFS, from some vertex
    visited = {}
    def dfs(cur_node):
       if cur_node in visited:
            return
        visited[cur_node] = 1
        for neighbor in adj[cur_node]:
           dfs(neighbor)
   ```
   3. 关于在DFS使用邻接矩阵或者邻接表的复杂度：
      1. 如果使用邻接矩阵，则复杂度为$O(V^2)$，因为每次选择新的点，都会遍历所有邻居。
      2. 如果使用邻接表，则复杂度为$O(E+V)$, 因为我们检查了所有的边和点。
      3. 总结：如果是稀疏图，则使用邻接表，否则邻接矩阵。
   4. m*n的矩阵，每个entry代表一个点，则复杂度为O(m*n), 因为每个点只有常数个边，访问邻居只需要常数时间（区别于邻接矩阵）
3. 关于dfs和bfs，拓扑排序的模板，见 https://www.techinterviewhandbook.org/algorithms/graph/
   - 关于BFS的visited判定，**需要注意的是需要在从queue中pop点之后立刻检查**，而不是仅仅放在枚举下一个点时检查（这在DFS中可行），这是因为BFS中会出现当前节点把位置相邻，且还在queue中的后续点再次入队的情况。考虑例子：`[[0 ,1], [1, 1]]`
5. 对于DFS，每次新开始需要重新call一次DFS，而**对于BFS，如果有多个起点，把他们同时放进开始的queue即可。**
6. 关于BFS和DFS的区别：
   1. BFS适用于在unweighted graph中寻找A到B的**最短路径**
   2. DFS适用于寻找**所有路径**，不保证找到最短路径。
7. 关于拓扑排序 (https://leetcode.com/problems/course-schedule/description/)：
   1. DFS算法： DFS每个未被visit的节点，backtrack时把当前节点加入拓扑排序尾部 (**stack**)。这个办法实现简单，但是**无法检测环**。https://www.youtube.com/watch?v=eL-KzMXSXXI
   2. Kahn算法：基于入度。
      1. 计算所有点的入度
      2. 挑选0入度的入列
      3. 队列非空时：
         1. 取出队首元素，加入topsort ans list
         2. 对于队首元素连接的点，入度-1
         3. 如果这些点入度变为了0，入队
      4. ans list则为答案，如果过短说明有环。

## Heap (Python heapq)
0. 关于heap的实现：https://www.youtube.com/watch?v=t0Cq6tVNRBA
1. 关于heapq的常见操作，注意：Python只提供小根堆：
    ```
    heapq.heapify(iterable) # 初始化小根堆, 不需要写sth = heapq.heapify(iterable), 这个完成之后，iterable自动变成了一个heap
    # 当heapify一个自定class的列时，可以override class内的 __lt__函数，说明如何判断less than.
    # e.g. return self.dis < other.dis

    heapq.nsmallest(k, iterable) # 前k小值
    heapq.nlargest(k, iterable) # 前k大值
    heapq.heappush(iterable, item) # 添加元素
    heapq.heappop(iterable) # pop root
    # 关于前k小，前k大，可以传入可选参数lambda函数key说明如何排序
    # e.g. heapq.nlargest(k, hash_tuples, key = lambda x: x[1])
    ```
2. 手工实现一个min heap:
   ```
   class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, index):
        return (index - 1) // 2

    def leftChild(self, index):
        return 2 * index + 1

    def rightChild(self, index):
        return 2 * index + 2

    def hasParent(self, index):
        return self.parent(index) >= 0

    def hasLeftChild(self, index):
        return self.leftChild(index) < len(**self**.heap)

    def hasRightChild(self, index):
        return self.rightChild(index) < len(self.heap)

    def swap(self, indexOne, indexTwo):
        self.heap[indexOne], self.heap[indexTwo] = self.heap[indexTwo], self.heap[indexOne]

    def insert(self, item):
        self.heap.append(item)
        self.heapifyUp(len(self.heap) - 1)

    def heapifyUp(self, index):
        while self.hasParent(index) and self.heap[self.parent(index)] > self.heap[index]:
            self.swap(self.parent(index), index)
            index = self.parent(index)

    def removeMin(self):
        if len(self.heap) == 0:
            raise Exception("Heap is empty")
        if len(self.heap) == 1:
            return self.heap.pop()
        minItem = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapifyDown(0)
        return minItem

    def heapifyDown(self, index):
        while self.hasLeftChild(index):
            smallerChildIndex = self.leftChild(index)
            if self.hasRightChild(index) and self.heap[self.rightChild(index)] < self.heap[smallerChildIndex]:
                smallerChildIndex = self.rightChild(index)
            if self.heap[index] < self.heap[smallerChildIndex]:
                break
            else:
                self.swap(index, smallerChildIndex)
            index = smallerChildIndex

   ```

## Trie
0. 关键词：**prefix**, **Lexicographical**
1. 实现一个Trie: https://leetcode.com/problems/implement-trie-prefix-tree/

## Interval
0. https://www.techinterviewhandbook.org/algorithms/interval/
1. 判断区间重叠:
    ```
    def is_overlap(a, b): 
      # ([)]
      return a[0] < b[1] and b[0] < a[1]
    ```
    需要注意的是，上述的区间重叠检测了所有情况的区间重叠，也即：`[(]), ([)], ([]), [()]`
    在某些时候（e.g. [Leetcode253](https://leetcode.com/problems/meeting-rooms-ii/)），当我们按区间起点sort后，我们并不需要这么强的检测方式。
    
3. 合并区间：
   ```
   def merge_intervals(a, b):
      return (min(a[0], b[0]), max(a[1], b[1]))
   ```
   需要注意的是这里应该取min/max, 而不是单纯取a[0]和b[1]

## DP
0. https://www.techinterviewhandbook.org/algorithms/dynamic-programming/
1. **01背包问题**：给定一系列物品，每个物品有价值和重量。背包有承重上限，问能获取的最大价值是？
   ```
   def solve(weights, values, max_weight):
      # f[i][w]: 前i个物品，背包容量为w时最大价值
      len_item = len(weights)
      f = [[0 for _ in range(max_weight + 1)] for _ in range(len_item + 1)]
      # for i in range(len_item + 1):
      #     f[i][0] = 0
      for i in range(1, len_item + 1):
          for w in range(max_weight + 1):
              if w < weights[i - 1]:
                  f[i][w] = f[i - 1][w]
              else:
                  f[i][w] = max(f[i - 1][w], f[i - 1][w - weights[i - 1]] + values[i - 1])
      return f[len_item][max_weight]
   ```
   递归版本（memorization）
   ```
   def solve(weights, values, max_weight):
      len_item = len(weights)
      def search(cur_weight, idx, memo):
          # base case handling
          if idx >= len_item or cur_weight <= 0:
              return 0
          # read from memo if exists
          if (cur_weight, idx) in memo:
              return memo[(cur_weight, idx)]
          # try possible item at index idx
          notake_item = search(cur_weight, idx + 1, memo)
          take_item = 0
          if cur_weight >= weights[idx]:
              take_item = search(cur_weight - weights[idx], idx + 1, memo) + values[idx]    
          memo[(cur_weight, idx)] = max(take_item, notake_item)
          return memo[(cur_weight, idx)]
   ```
   相关问题：[Leetcode416](https://leetcode.com/problems/partition-equal-subset-sum/description/)

## Binary
1. 从十进制转化为二进制：
    ```
    decimal_num = 10
    bianry_string = bin(decimal_num)
    ```
2. 从二进制转换为十进制：
    ```
    binary_string = '0b1010'
    decimal_number = int(binary_string, 2)
    ```
3. 二进制前缀：`0b`
## Math
 0. https://www.techinterviewhandbook.org/algorithms/math/
 1. 左移变大，右移变小：`1<<x` = `2^x`。注意`<<`的优先级低，如果想要表示`2^x  -1`则应该写`(1<<x) - 1`

## Geometry
0. https://www.techinterviewhandbook.org/algorithms/geometry/
1. 判断矩阵重叠：
    ```
    overlap = rect_a.left < rect_b.right and \
    rect_a.right > rect_b.left and \
    rect_a.top > rect_b.bottom and \
    rect_a.bottom < rect_b.top
   ```

## Union Find Set
1. 并查集的实现：
   ```
      class UnionFind:
          # 给定一个nums list，把其变成一个并查集
          # 还实现了寻找一个set的size的功能 (`self.count`)
          def __init__(self, nums):
              self.parent = {}
              self.size = {}
              self.rank = {}
              for num in nums:
                  self.parent[num] = num
                  self.size[num] = 1
                  self.rank[num] = 1
          
          def find(self, node):
              if self.parent[node] == node:
                  return node
              # 路径压缩 
              self.parent[node] = self.find(self.parent[node])
              return self.parent[node]
      
          def union(self, node_0, node_1):
              # union总是对root进行操作！
              root_0 = self.find(node_0)
              root_1 = self.find(node_1)
              if root_0 == root_1:
                  return
              if self.rank[root_0] > self.rank[root_1]:
                  self.parent[root_1] = root_0
                  self.size[root_0] += self.size[root_1]
              elif self.rank[root_0] < self.rank[root_1]:
                  self.parent[root_0] = root_1
                  self.size[root_1] += self.size[root_0]
              else:
                  self.parent[root_1] = root_0
                  self.size[root_0] += self.size[root_1]
                  self.rank[root_0] += 1
          
          def count(self, node):
              return self.size[self.find(node)]
   ```
   使用并查集：
   ```
   nums = [1, 3, 5, 6, 9]
   union_find_nums = UnionFind(nums)
   union_find_nums.find(1) # 1
   union_find_nums.find(3) # 3
   union_find_nums.count(1) # 1
   union_find_nums.union(1, 3)
   union_find_nums.find(3) # 1
   union_find_nums.count(1) # 2
   ```
2. 处理归类，等价等问题的时候（通常是在图的问题里）。在非图的问题里，这样的题目是比较强的提示：（[这个题](https://leetcode.com/problems/longest-consecutive-sequence/description/)）

      - 有明确的sub-set定义，这样的定义与order无关
      - 需要count numbers in sub set


## Bitwise Manipulations
### XOR:
1. 关于XOR运算的性质：
  - **可交换**：A ^ B = B ^ A
  - **可结合**：A ^ (B ^ C) = (A ^ B) ^ C
  - **Identity element**: A ^ 0 = A
  - **自反**：A ^ A = 0
2. 由以上性质可以导出的结论：
  - **A ^ B = C** -> A ^ B ^ B = C ^ B -> **A = C ^B**
  - 于是：`arr[i] ^ arr[i+1] ^ ... ^ arr[j] = prefix[j] ^ prefix[i-1]`. **可以用前缀和来计算XOR区间和。**

## pandas
1. 创建一个df:
   ```
   data = [[row_1], [row_2], ..., [row_m]]
   df = pd.DataFrame(data, columns = ['col_1', ..., 'col_n'])
   ```
   或者：
   ```
   df = pd.DataFrame({'col_1': [data_in_col_1], 'col_2': [data_in_col_2]}, index = [equal_len_of_data])
   ```
   来创建多个行，或者：
   ```
   df = pd.DataFrame({'col_1': data, 'col_2': data}, index=[0])
   ```
   来创建一行，注意这里字典里面的value不是list
   
2. Append row to existing df
   ```
   new_row = pd.DataFrame({'col_1': data, ..., 'col_n': data}, index = [0])
   pd.concat([original_df, new_row]).reset_index(drop=True)
   ```
   
3. filter by condition
   ```
   filtered_df = df[df['col_name'] > some_value]
   ```


## 关于Python语法
1. `nonlocal` / `global`:

    - `nonlocal` is used in **nested functions**, when a inner function wants to reference an **immutable** variable outside of its scope.
    - `global` is used for **global variables**, when a function wants to change its value, the `global` keyword should be used.
    - `nonlocal` applied to global variables will cause Error.
    - If a variable is **mutable** e.g. a list, then `nonlocal` and `global` are not necessary.

## General Notices
1. 在循环中，如果循环结束条件是可变元素（dict, deque, list, ...），要注意这个元素是否被改变(**del**, **pop**, **popleft**, ...)，如果在循环过程中这个被改变，那么可能产生以外的效果。
2. 尤其是在使用deque, dict, list的时候，当心这个可变元素在其他地方被修改之后导致的element missing, empty deque/list的问题。
