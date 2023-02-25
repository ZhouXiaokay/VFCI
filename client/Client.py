import time
import grpc
import pickle
import transmission.pickle.aggregate_server_pb2 as aggregate_server_pb2
import transmission.pickle.aggregate_server_pb2_grpc as aggregate_server_pb2_grpc


class Client:

    def __init__(self, server_address, client_rank):
        self.server_address = server_address
        self.client_rank = client_rank

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = aggregate_server_pb2_grpc.AggregateServerServiceStub(channel)

    def __bottom_dumps(self, plain_vector, epoch, rnd):
        # print(">>> client bottom transmit start")

        # print("size of msg: {} bytes".format(sys.getsizeof(enc_vector.serialize())))

        # create request
        request_start = time.time()
        request = aggregate_server_pb2.bottom_output(
            client_rank=self.client_rank,
            round=rnd,
            epoch=epoch,
            params_msg=pickle.dumps(plain_vector)
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        # print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.middle_fp_bp(request)
        comm_time = time.time() - comm_start

        # load grad tensor

        assert self.client_rank == response.client_rank

        bottom_grad_vector = pickle.loads(response.grad_msg)
        return bottom_grad_vector

    def transmit(self, plain_vector, epoch, rnd):
        trans_start = time.time()
        # received:list, received tensors convert received to tensors
        # print(">>> client transmission cost {:.2f} s".format(time.time() - trans_start))

        received = self.__bottom_dumps(plain_vector, epoch, rnd)

        return received


if __name__ == '__main__':
    serv_address = "127.0.0.1:59000"
    # ctx_file = "../../transmission/ts_ckks.config"
    # client_rank = 0
    #
    # client = Client(serv_address, client_rank, ctx_file)
