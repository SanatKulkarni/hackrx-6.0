class Node:
    def __init__(self,data):
        self.data = data 
        self.next = None 

class LinkedList:
    def __init__(self):
        self.head = None 

    def append(self,data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head 
            while current.next:
                current = current.next 
            current.next = new_node

    def delete(self,key):
        if self.head is None:
            return 
        
        if self.head.data == key:
            self.head = self.head.next 
            return 

        current = self.head 
        while current.next:
            if current.next.data == key:
                current.next = current.next.next 
                return 
            current = current.next 

    def print_list(self):
        current=self.head 
        while current:
            print(current.data,end=' ')
            current=current.next 
        print()
        
    

if __name__ == "__main__":
    my_list = LinkedList()
    my_list.append(1)
    my_list.append(2)
    my_list.append(3)
    my_list.append(4)

    print("Original Linked List")
    my_list.print_list()

    my_list.delete(2)
    print("List after deleting 2")
    my_list.print_list()