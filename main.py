import jax
import jax.numpy as jnp
import numpy as np

def compress(row, size):
    nonzero = row != 0
    order = jnp.argsort(jnp.where(nonzero, 0, 1))
    return row[order]

def merge(row, size):
    row = compress(row, size)

    result = jnp.zeros_like(row)
    score = jnp.array(0, dtype=row.dtype)
    outix = jnp.array(0)
    skip = jnp.array(False)

    def body(i, state):
        result, score, outix, skip = state

        current = row[i]

        mergedd = (
            (i + 1 < size)
            & (current != 0)
            & (current == row[i + 1])
            & (~skip)
        )

        def do_merge(state):
            result, score, outix, skip = state

            value = current * 2

            result = result.at[outix].set(value)
            score = score + value
            outix = outix + 1

            return result, score, outix, jnp.array(True)

        def do_normal(state):
            result, score, outix, skip = state

            def write(state):
                result, score, outix, skip = state
                result = result.at[outix].set(current)
                return result, score, outix + 1, jnp.array(False)

            def nothing(state):
                result, score, outix, skip = state
                return result, score, outix, jnp.array(False)

            return jax.lax.cond(
                current != 0,
                write,
                nothing,
                state
            )

        def skipped(state):
            result, score, outix, skip = state
            return result, score, outix, jnp.array(False)

        return jax.lax.cond(
            skip,
            skipped,
            lambda state: jax.lax.cond(
                mergedd,
                do_merge,
                do_normal,
                state
            ),
            state
        )

    result, score, outix, skip = jax.lax.fori_loop(
        0,
        size,
        body,
        (result, score, outix, skip)
    )

    return result, score

def leftM(board,size):
  rows, scores = jax.vmap(merge,in_axes=(0,None))(board,size)
  return rows, jnp.sum(scores)

def move(board, action, size):
  def up(b):
    new, score = leftM(b.T,size)
    return new.T, score

  def right(b):
    new, score = leftM(jnp.flip(b, axis=1),size)
    return jnp.flip(new, axis=1), score

  def down(b):
    new, score = leftM(jnp.flip(b.T, axis=1),size)
    return jnp.flip(new, axis=1).T, score

  def left(b):
    return leftM(b,size)

  return jax.lax.switch(action,[up, right, down, left],board)

def add(board,key):
  empty = board == 0

  k1, k2 = jax.random.split(key)
  randomV = jax.random.uniform(k1,shape=board.shape)

  priorities = jnp.where(empty,randomV,-1.0)

  index = jnp.argmax(priorities)
  row = index // 4
  col = index % 4
  value = jnp.where(jax.random.uniform(k2) < 0.9,2,4)

  def a(aa): return aa.at[row,col].set(value)
  def b(bb): return bb
  return jax.lax.cond(jnp.any(empty),a,b,board)

def gameOver(board, size):
    has_empty = jnp.any(board == 0)
    horizontal = jnp.any(board[:, :-1] == board[:, 1:])
    vertical = jnp.any(board[:-1, :] == board[1:, :])
    return (~has_empty) & (~horizontal) & (~vertical)

def neuraln(network,x,bests):
  w1,b1,w2,b2,w3,b3,w4,b4 = network
  x1 = jax.nn.one_hot(jnp.log2(jnp.maximum(x.flatten(),1)).astype(jnp.int32),bests).astype(jnp.float32).flatten()
  x = jnp.log2(jnp.maximum(x.flatten(),1)).astype(jnp.int32)
  x = x/bests
  x = jax.nn.relu(jnp.matmul(w1.T,x)+b1)
  x = jax.nn.relu(jnp.matmul(w2.T,x)+b2)
  x = jax.nn.relu(jnp.matmul(w3.T,x)+b3)
  x = jnp.matmul(w4.T,x)+b4
  return jnp.arange(4)[jnp.argsort(x)][::-1]

def getNetwork(key,size,bests):
  inputSize = size**2
  hidden1 = 64
  hidden2 = 64
  hidden3 = 32
  outputSize = 4
  k1,k2,k3,k4,k5,k6,k7,k8 = jax.random.split(key,8)

  w1 = jax.random.normal(k1,(inputSize,hidden1))*0.1
  b1 = jax.random.normal(k4,(hidden1,))*0.1
  w2 = jax.random.normal(k2,(hidden1,hidden2))*0.1
  b2 = jax.random.normal(k5,(hidden2,))*0.1
  w3 = jax.random.normal(k3,(hidden2,hidden3))*0.1
  b3 = jax.random.normal(k6,(hidden3,))*0.1
  w4 = jax.random.normal(k7,(hidden3,outputSize))*0.1
  b4 = jax.random.normal(k8,(outputSize,))*0.1

  return w1,b1,w2,b2,w3,b3,w4,b4

def getBoard(key,size):
 board = jnp.zeros((size,size),dtype=jnp.int32)
 k1,k2 = jax.random.split(key)
 board = add(board,k1)
 board = add(board,k2)
 return board

def play(network,board,key,size,bests):
  board0 = board.copy()
  first = jnp.array(True)
  score = 0
  def condition(state):
    return jnp.logical_not(jnp.logical_and(jnp.logical_not(state[4]),gameOver(state[1],size)))
  def body(state):
    board0,board,score,key,first = state
    board0 = board.copy()
    moveR = neuraln(network,board,bests)
    index = 0
    def cond(state):
      index,board,score,key,scoreM = state
      return jnp.logical_and(index<4,jnp.all(board0==board))
    def body(state):
       index,board,score,key,scoreM = state
       board, scoreM = move(board,moveR[index],size)
       index += 1
       return index,board,score,key,scoreM
    index,board,score,key,scoreM = jax.lax.while_loop(
      cond,body,(index,board,score,key,0)
    )
    score += scoreM
    key,k2 = jax.random.split(key)
    board = add(board,k2)
    return board0,board,score,key,jnp.array(False)
  return jax.lax.while_loop(
      condition,body,(board0,board,score,key,first)
  )

def a2(p1,p2,key):
  keys = jax.random.split(key,len(p1))
  p3 = []
  p4 = []
  for i in zip(p1,p2,keys):
    pp1,pp2,k = i
    r = jax.random.bernoulli(k,0.5,pp1.shape)
    pp3 = jnp.where(r,pp1,pp2)
    pp4 = jnp.where(jnp.logical_not(r),pp1,pp2)
    p3.append(pp3)
    p4.append(pp4)
  return tuple(p3),tuple(p4)

def a3(p,key):
  k1,k2 = jax.random.split(key,2)
  keys1 = jax.random.split(k1,len(p))
  keys2 = jax.random.split(k2,len(p))
  result = []
  for i in zip(p,keys1,keys2):
    pp1,k1,k2 = i
    r = jax.random.uniform(k1,pp1.shape)<pMut
    result.append(pp1+jax.random.normal(k2,pp1.shape)*((pp1.max()-pp1.min())+0.0001)/100*r)
  return tuple(result)

size = 4
bests = 14
populationSize = 100
nSteps = 280
pMut = 0.001
key = jax.random.key(0)
k1,k2,k3,k4 = jax.random.split(key,4)

keys = jax.random.split(k1,populationSize)
population = jax.vmap(getNetwork,in_axes=(0,None,None))(keys,size,bests)
keys = jax.random.split(k2,populationSize)
boards = jax.vmap(getBoard,in_axes=(0,None))(keys,size)
scores = jnp.zeros(populationSize,dtype=jnp.int32)

ybest = -jnp.inf
keysSteps = jax.random.split(k3,nSteps)
for i in range(nSteps):
  kk1,k1,k2,k3 = jax.random.split(keysSteps[i],4)
  kkeys = jax.random.split(kk1,4)
  kkeys1 = jax.random.split(kkeys[0],populationSize)
  kkeys2 = jax.random.split(kkeys[1],populationSize)
  kkeys3 = jax.random.split(kkeys[2],populationSize)
  kkeys4 = jax.random.split(kkeys[3],populationSize)
  keys = jax.random.split(k1,4)
  keys1 = jax.random.split(keys[0],populationSize)
  keys2 = jax.random.split(keys[1],populationSize)
  keys3 = jax.random.split(keys[2],populationSize)
  keys4 = jax.random.split(keys[3],populationSize)
  boards = jax.vmap(getBoard,in_axes=(0,None))(kkeys1,size)
  board0,board,y1,key,first = jax.vmap(play,in_axes=(0,0,0,None,None))(population,boards,keys1,size,bests)
  boards = jax.vmap(getBoard,in_axes=(0,None))(kkeys2,size)
  board0,board,y2,key,first = jax.vmap(play,in_axes=(0,0,0,None,None))(population,boards,keys2,size,bests)
  boards = jax.vmap(getBoard,in_axes=(0,None))(kkeys3,size)
  board0,board,y3,key,first = jax.vmap(play,in_axes=(0,0,0,None,None))(population,boards,keys3,size,bests)
  boards = jax.vmap(getBoard,in_axes=(0,None))(kkeys4,size)
  board0,board,y4,key,first = jax.vmap(play,in_axes=(0,0,0,None,None))(population,boards,keys4,size,bests)
  y = jnp.mean(jnp.stack([y1,y2,y3,y4],axis=0),axis=0)
  if jnp.max(y) > ybest:
    ybest = jnp.max(y)
    def best(x): return x[jnp.argmax(y)]
    xbest = jax.tree.map(best,population)
  print("{}: {}".format(i,jnp.max(y)))

  newPopulation = []
  for i2 in range(populationSize-2):
    r1,r2 = np.random.choice(populationSize,(2,),replace=False)
    newPopulation.append(r1 if y[r1] > y[r2] else r2) 
  newPopulation = jnp.array(newPopulation)
  def filters(x): return x[newPopulation]
  population = jax.tree.map(filters,population)

  #def first(x): return x[:populationSize//2-1]
  #def second(x): return x[populationSize//2-1:]
  #p1 = jax.tree.map(first,population)
  #p2 = jax.tree.map(second,population)
  #keys2 = jax.random.split(k2,populationSize//2-1)
  #p1,p2 = jax.vmap(a2)(p1,p2,keys2)
  #def pair(x1,x2):
     #return jnp.concatenate([x1,x2],axis=0)
  #population = jax.tree.map(pair,p1,p2)

  def insert(x,best): return jnp.concatenate([x,best[None]],axis=0)
  population = jax.tree.map(insert,population,xbest)
  keys3 = jax.random.split(k3,populationSize-1)
  population = jax.vmap(a3)(population,keys3)
  population = jax.tree.map(insert,population,xbest)