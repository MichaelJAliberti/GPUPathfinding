// By Dmitry Zimin and Michael Aliberti

#ifndef BINARY_HEAP_H_
#define BINARY_HEAP_H_

using namespace std;

const int MAX_SIZE = 100000; //the maximum amount of elements our heap should have.

template <typename Object>
class Heap
{
public:
   Heap(){
      elements = 0;
   };
   void insert(Object* item){// Add the object pointer item to the heap
		
		if (elements >= MAX_SIZE){
			cout << "Heap is full; can't insert "<< endl;
			return;
		}

		// FILLED IN CODE

		// Insert item as rightmost leaf
		item->position = elements;
		array[elements] = item;
		array[elements]->position = elements;

		// Update number of elements in heap
		elements++;

		// Upheap until min heap property is restored
		upHeap(item->position);

		return;
   }; 

   Object* remove_min(){
      if (elements == 0){
		   cout << "empty heap error, can't delete" << endl;
		}
		
		// FILLED IN CODE

		// Swap rightmost element and root
		Object* temp = this->array[0];
		this->array[0] = this->array[elements-1];

		// Update positions
		this->array[0]->position = 0;
		temp->position = -1;

		// Delete rightmost element (original root) and adjust element count				
		this->array[elements-1] = NULL;
		this->elements--;

		// Downheap new root until minHeap property is restored
		downHeap(0);

		return temp;
   };       // Remove the smallest element in the heap & restructure heap
   
   void decreaseKey(int pos, int val)	// Decreases Key in pos to val
   {
		// FILLED IN CODE

		this->array[pos]->key = val;	// change key to the new value
		upHeap(pos);						// upHeap until minHeap property is restored
		return;
   }; 
   

   bool IsEmpty() const {  return (elements <= 0);};
   bool IsFull() const {return (elements >=MAX_SIZE );};
   int count() const {return elements;};
   Object* value(int pos) const{ //return a pointer to an object in heap position
	   if (pos >= elements){
		   cout << "Out of range of heap " << pos << "elements " << elements << endl;
	   }
      return (array[pos]);
   };  
protected:
   Object* array[MAX_SIZE];
   int elements;       //  how many elements are in the heap
private:
   void downHeap(int pos){	// starting with element in position pos, sift it down the heap 
									// until it is in final min-heap position
					   
		// FILLED IN CODE

		// Check if pos is within bounds
		if (pos >= this->elements)
			return;
					   		   
		Object* item = this->array[pos];					
		Object* temp;
		int childPos = 2*pos+1; 		// left child
		
		// Check if left child exists
		if (childPos >= this->elements) return;
		if (this->array[childPos] == NULL) return;

		// Check if right child exists
		if (childPos+1 < this->elements)
			if (this->array[childPos+1] != NULL)
				// Change comparison target to right child if it has lower key
				if(this->array[childPos]->key > this->array[childPos+1]->key)
					childPos += 1;
		
		// If parent has larger key than smallest child, swap parent and child
		if(item->key > this->array[childPos]->key){

			// Swap parent and child
			temp = this->array[childPos];
			this->array[childPos] = item;
			this->array[pos] = temp;

			// Upadate positions
			this->array[pos]->position = pos;
			this->array[childPos]->position = childPos;
			
			// Recur until original node is downheaped to leaf or is larger than child
			downHeap(childPos);
		} 
		else
			return;
   }; 

   void upHeap(int new_pos){	// starting with element in position int, sift it up the heap
                       			// until it is in final min-heap position
					   
		// FILLED IN CODE

   	// Return if upheaped to root
		if(new_pos == 0)
			return;

		Object* item = this->array[new_pos];
		Object * temp;
		int parentPos = (new_pos-1)/2;				// poistion of parent
		
		if(item->key < this->array[parentPos]->key)		// check if child is smaller then parent
		{														// if yes then swap the two and upHeap
			
			// Swap parent and child
			temp = this->array[parentPos];
			this->array[parentPos] = item;
			this->array[new_pos] = temp;
			
			// Update positions
			this->array[new_pos]->position = new_pos;
			this->array[parentPos]->position = parentPos;
			
			// Recur until original node is upheaped to root or its parent is smaller
			upHeap(parentPos);
		} 
		else
			return;
      
   };   
};

#endif